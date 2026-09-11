#!/usr/bin/env python3
"""CLI for gallery construction and ProtectReID inference."""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch

from protectreid.config import ProtectConfig
from protectreid.data import image_id, list_images, load_image, save_image
from protectreid.gallery_builder import build_gallery
from protectreid.models import E4EEncoder, IdentityExtractor, load_generator, load_reid_extractor
from protectreid.pipeline import ProtectReIDPipeline
from protectreid.retrieval import LatentGallery


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generator", required=True, help="StyleGAN3 G_ema pickle")
    parser.add_argument("--reid", required=True, help="released ResNet50 re-ID checkpoint")
    parser.add_argument("--stylegan-root", default=None, help="official stylegan3 repo root")
    parser.add_argument("--device", default="auto")


def cmd_build_gallery(args: argparse.Namespace) -> None:
    device = _device(args.device)
    reid = IdentityExtractor(load_reid_extractor(args.reid, device=device))
    e4e = E4EEncoder.load(args.e4e, args.e4e_root, device=device)
    gallery = build_gallery(args.images, args.output, reid, e4e, image_size=args.image_size)
    print(f"saved {len(gallery.ids or [])} gallery entries to {args.output}")


def cmd_protect(args: argparse.Namespace) -> None:
    device = _device(args.device)
    generator = load_generator(args.generator, device=device, stylegan_root=args.stylegan_root)
    reid = IdentityExtractor(load_reid_extractor(args.reid, device=device))
    gallery = LatentGallery.load(args.gallery, device=device)
    config = ProtectConfig(
        retrieval_k=args.k,
        refinement_steps=args.steps,
        step_size=args.alpha,
        visual_margin=args.beta,
        attention_temperature=args.temperature,
        reciprocal_epsilon=args.reciprocal_eps,
        reciprocal_clip=args.reciprocal_clip,
    )
    pipeline = ProtectReIDPipeline(generator, reid, gallery, config, device=device)
    paths = list_images(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = int(getattr(generator, "img_resolution", args.image_size))
    metadata_path = output_dir / "metadata.jsonl"

    with metadata_path.open("w", encoding="utf-8") as metadata:
        for path in paths:
            original = load_image(path, resolution).unsqueeze(0)
            sample_id: Optional[str] = image_id(path) if args.exclude_source else None
            query_ids = [sample_id] if sample_id is not None and gallery.ids is not None else None
            result = pipeline.protect(original, query_ids=query_ids)
            stem = path.stem
            image_path = output_dir / f"{stem}.png"
            latent_path = output_dir / f"{stem}.pt"
            save_image(result.protected_images, image_path)
            torch.save(result.final_latents[0].detach().cpu(), latent_path)
            record = {
                "input": str(path),
                "output": str(image_path),
                "latent": str(latent_path),
                "top_indices": result.top_indices[0].detach().cpu().tolist(),
                "bottom_indices": result.bottom_indices[0].detach().cpu().tolist(),
                "top_scores": result.top_scores[0].detach().cpu().tolist(),
                "bottom_scores": result.bottom_scores[0].detach().cpu().tolist(),
            }
            metadata.write(json.dumps(record) + "\n")
            print(f"protected {path} -> {image_path}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ProtectReID reference implementation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gallery = subparsers.add_parser("build-gallery", help="encode the Market-1501 retrieval gallery")
    gallery.add_argument("--images", required=True)
    gallery.add_argument("--output", required=True, help="output .npz")
    gallery.add_argument("--reid", required=True)
    gallery.add_argument("--e4e", required=True, help="e4e checkpoint")
    gallery.add_argument("--e4e-root", required=True, help="released ProtectReID/e4e code root")
    gallery.add_argument("--image-size", type=int, default=256)
    gallery.add_argument("--device", default="auto")
    gallery.set_defaults(func=cmd_build_gallery)

    protect = subparsers.add_parser("protect", help="protect one image or a directory")
    protect.add_argument("--input", required=True)
    protect.add_argument("--output", required=True)
    protect.add_argument("--gallery", required=True, help="official .mat or generated .npz gallery")
    _add_model_args(protect)
    protect.add_argument("--image-size", type=int, default=256)
    protect.add_argument("--k", type=int, default=10)
    protect.add_argument("--steps", type=int, default=10)
    protect.add_argument("--alpha", type=float, default=1e-3)
    protect.add_argument("--beta", type=float, default=0.5)
    protect.add_argument("--temperature", type=float, default=0.1)
    protect.add_argument("--reciprocal-eps", type=float, default=None)
    protect.add_argument("--reciprocal-clip", type=float, default=None)
    protect.add_argument(
        "--exclude-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exclude the query's same-stem gallery row when gallery ids are present",
    )
    protect.set_defaults(func=cmd_protect)
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    args.func(args)
