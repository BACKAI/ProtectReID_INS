# ProtectReID: Identity Retrieval and Hierarchical Latent Code Protection

This directory is a clean implementation of **ProtectReID**, the method in
the accepted Elsevier *Information Sciences* paper:

> Seung-hyeok Back and Seok Bong Yoo, “Privacy-preserving person
> re-identification through identity retrieval and hierarchical latent code
> protection,” *Information Sciences*, volume 756, article 123803, 2026.

The implementation follows the paper's three-stage inference pipeline:

1. retrieve identity-aligned top-k and structurally dissimilar bottom-k W+
   codes from a fixed gallery;
2. apply parameter-free reciprocal self-attention to the top-k codes and
   mean-pool the bottom-k codes;
3. refine the resulting W+ code for each input image using the visual-margin
   objective on coarse layers and the identity-cosine objective on fine layers.

The original `ProtectReID_INS-main` directory is not imported or modified.
The code here is intentionally self-contained at the algorithm level and
accepts the official StyleGAN3/e4e/re-ID checkpoints through explicit command
line arguments.

## Important training/inference distinction

ProtectReID does **not** train a protector network in the reported method.
StyleGAN3, e4e, and the re-ID extractor are frozen off-the-shelf models. The
only optimized parameters are the 14 x 512 W+ values of each individual image
during inference. “Training” in this repository therefore means constructing
the offline identity-to-latent gallery. The paper reports ten per-image
refinement iterations, so inference is iterative and is not a single forward
pass.

The recovery networks described in the paper are attacker-side evaluation
models, not components of the defense. To reproduce that experiment, export
the protected training split with this implementation and train the official
Restormer setup on paired `(protected image, original image)` samples; keep
the official test split for evaluation only.

## Paper configuration implemented here

| Component | Default | Implementation |
| --- | ---: | --- |
| W+ shape | `14 x 512` | StyleGAN3 generator with 14 style inputs |
| Retrieval gallery | Market-1501 gallery | 19,732 gallery images in the paper |
| Retrieval count | `k = 10` | top-10 and bottom-10 |
| Similarity | cosine | normalized identity features + FAISS `IndexFlatIP` when available |
| Source exclusion | enabled when IDs are available | removes the exact source row, not every image of the same identity |
| Reciprocal attention | `1 / (QK^T / 512)` | row-wise softmax after reciprocal transformation |
| Attention temperature | `0.1` | exposed as `--temperature` |
| Reciprocal epsilon | none | default is the paper's no-epsilon formulation |
| Softmax clipping | none | default is the paper's no-clipping formulation |
| Coarse layers | `0, 1, 2` | visual-margin gradient update |
| Fine layers | `3 ... 13` | identity-cosine gradient update |
| Refinement iterations | `T = 10` | exposed as `--steps` |
| Step size | `alpha = 0.001` | exposed as `--alpha` |
| Visual margin | `beta = 0.5` | exposed as `--beta` |
| Generator noise | constant | `noise_mode='const'` |

The visual objective is

```text
L_vis = max(0, ||G(f*) - I_b||_2^2 - ||G(f*) - I_o||_2^2 + beta)
```

and is back-propagated only to W+ layers 0–2. The identity objective is

```text
L_id = 1 - cosine(E(G(f*)), E(I_o))
```

and is back-propagated only to W+ layers 3–13. Each iteration performs the
coarse update first, synthesizes the updated code, and then performs the fine
identity update, as in Algorithm 3 of the paper.

## Datasets and evaluation protocol

The paper evaluates on the following person re-identification benchmarks:

| Dataset | Images / boxes | Identities | Cameras / notes |
| --- | ---: | ---: | --- |
| Market-1501 | 32,668 pedestrian images | 1,501 | six surveillance cameras |
| MSMT17 | 126,441 labeled boxes | 4,101 | 15 cameras, indoor and outdoor |
| CUHK03 | 14,097 detected boxes | 1,467 | cross-view re-ID benchmark |

The reported retrieval gallery is built **only from the 19,732 Market-1501
gallery images**. For every benchmark, only query images are protected; the
trusted evaluation galleries stay unprotected. In the cross-dataset setting,
the Market-1501 retrieval gallery remains fixed while MSMT17 or CUHK03 uses
its standard evaluation split.

For authorized matching, use the re-ID network trained on unprotected images
and compare protected queries against the unprotected trusted gallery. The
paper reports Rank-1 and mAP with AGW, BagTricks, and ABD-Net. For the
unauthorized-transfer analysis, protected queries are evaluated with unseen
OSNet, TransReID, and ReIDMamba models. For recovery resistance, the paper
trains Restormer using 90% of protected/original training pairs, uses 10% for
validation, and holds out the official test split.

Visual metrics are PSNR, SSIM, and LPIPS. Lower PSNR/SSIM and higher LPIPS
indicate stronger visual distortion. Recovery evaluation additionally reports
Rank-1 on the recovered images. The paper also evaluates BIRD and DPIR as
off-the-shelf restoration attackers and studies a surrogate retrieval-gallery
attacker.

## Checkpoints

Download the checkpoints linked by the paper's released repository and pass
their paths explicitly. The expected assets are:

| Asset | Role |
| --- | --- |
| StyleGAN3 `G_ema` pickle | synthesizes protected images from W+ codes |
| e4e checkpoint | encodes gallery images into W+ codes when building a gallery |
| Market-1501 ResNet50 re-ID checkpoint | extracts the 512-D identity feature |
| vector gallery (`Vector_gallery.mat`) | official paired `gallery_f` and `gallery_e4e` arrays |

The paper's code repository is
[BACKAI/ProtectReID_INS](https://github.com/BACKAI/ProtectReID_INS). Its
released model links are also listed in that repository. A prebuilt official
MATLAB gallery can be used directly for inference; a generated NPZ gallery
also stores image stems so exact source exclusion works automatically.

The StyleGAN3 pickle loader requires a checkout of
[NVLabs/stylegan3-ada-pytorch](https://github.com/NVlabs/stylegan3-ada-pytorch)
via `--stylegan-root`. The e4e gallery builder requires the root of the
released ProtectReID/e4e code via `--e4e-root`.

## Installation

Use a CUDA-enabled PyTorch installation compatible with the target GPU. Then:

```bash
cd /var/tmp/jnuadmin_storage/shback/z_paper/ProtectReID/ProtectReID_real
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the official StyleGAN3 dependencies in the StyleGAN3 checkout as
described by NVLabs. `faiss-cpu` is optional: the code falls back to an exact
torch matrix-product search if FAISS is unavailable. A CUDA FAISS build can be
used separately for large galleries.

## Building the identity-to-latent gallery

For faithful main experiments, use the official precomputed
`Vector_gallery.mat`. To build an equivalent NPZ gallery from aligned and
cropped Market-1501 gallery images:

```bash
python run.py build-gallery \
  --images /path/to/Market-1501/bounding_box_test \
  --output /path/to/Vector_gallery.npz \
  --reid /path/to/resnet50_reid.pth \
  --e4e /path/to/e4e_reid.pt \
  --e4e-root /path/to/ProtectReID_INS \
  --device cuda:0
```

Each entry is saved as:

```text
gallery_f    [N, 512]       identity feature
gallery_e4e  [N, 14, 512]   W+ latent code
ids          [N]            image filename stems
```

The gallery is an offline artifact. Do not rebuild it from protected images;
the paper pairs identity features and latent codes from the original gallery
images.

## Protecting images

Inputs should be RGB, aligned/cropped person images. They are resized to the
generator's square resolution for synthesis and internally resized to
`256 x 128` with ImageNet normalization for the ResNet50 re-ID extractor.

```bash
python run.py protect \
  --input /path/to/query_images \
  --output /path/to/protected_outputs \
  --generator /path/to/stylegan3_ada_reid.pkl \
  --gallery /path/to/Vector_gallery.mat \
  --reid /path/to/resnet50_reid.pth \
  --stylegan-root /path/to/stylegan3-ada-pytorch \
  --device cuda:0 \
  --k 10 \
  --steps 10 \
  --alpha 0.001 \
  --beta 0.5 \
  --temperature 0.1
```

For a generated NPZ gallery, source exclusion is enabled by default and uses
the query filename stem. The official `.mat` file does not necessarily carry
IDs; in that case the pipeline additionally checks for a near-exact identity
feature match and excludes that row when the query itself is present. For
cross-dataset queries, no exact row is found and the full gallery is used.

Outputs are:

```text
protected_outputs/
  image_name.png       protected image
  image_name.pt        final [14,512] W+ code
  metadata.jsonl       retrieval indices/scores and source/output paths
```

The implementation processes one query at a time so every query can have its
own gallery exclusion and so memory is predictable. Ten refinement iterations
require repeated StyleGAN synthesis and are intentionally not real-time at
large scale. The paper reports approximately 92.87 GFLOPs and 362.79 ms per
image on its RTX 3080 setup, with the cost dominated by latent refinement.

## Reproducing the recovery experiment

1. Protect the official training split and save the original/protected path
   pairs.
2. Split pairs 90/10 into attacker training/validation; never train on the
   official test split.
3. Train Restormer on protected-to-original pairs using its official
   configuration and select the lowest validation reconstruction loss.
4. Run the trained Restormer on the held-out protected test images.
5. Report PSNR, SSIM, LPIPS, and re-ID Rank-1 on recovered images.

The paper's primary attacker is supervised black-box oracle recovery: the
attacker can query the protection system and collect paired samples, but does
not receive generator parameters, W+ codes, or the authorized re-ID model.
This is an evaluation threat model, not a claim that all possible recovery or
cross-model attacks are impossible.

## Verification and implementation corrections

The old directory was inspected only as a reference. This directory corrects
the parts that did not match the paper's equations/algorithms:

- keeps all K retrieved W+ codes until reciprocal attention instead of
  averaging them before attention;
- retrieves bottom-k by the original cosine ranking and supports exact source
  row exclusion;
- uses `QK^T / 512`, not `QK^T / sqrt(512)`;
- removes the random channel permutation;
- uses coarse layers 0–2 and fine layers 3–13;
- uses the two sequential gradient updates in Algorithm 3, without
  MI-FGSM momentum or an `alpha/8` coarse step;
- keeps generator and extractor parameters frozen and optimizes only the
  per-image W+ tensor;
- removes hard-coded `.cuda()` calls and makes device/path handling explicit.

Run a syntax check without loading any checkpoints:

```bash
python -m compileall -q protectreid run.py
```

## Citation

Back, Seung-hyeok, and Seok Bong Yoo. “Privacy-preserving person
re-identification through identity retrieval and hierarchical latent code
protection.” *Information Sciences*, vol. 756, 2026, article 123803.
https://doi.org/10.1016/j.ins.2026.123803
