# PAAS_v4 — Geometric Semantic Decoupling (GSD)

A focused, faithful implementation of **Geometric Semantic Decoupling** (arXiv 2603.09242) for
AI-generated / deepfake image detection. GSD removes the dominant *semantic* directions of a frozen
CLIP feature space by orthogonal projection, forcing the detector to learn forgery cues in the
**semantic null-space** — no auxiliary disentanglement loss required.

Runs on the global `python3.12` / `transformers==4.37.2` env. **Standalone**: the CLIP ViT-L/14-336
weights (`base_models/`) and the label resolver (`gsd/get_label.py`) are vendored into this project —
no external code or model paths are required (only the image datasets live outside).

## Classification: 3-class real / pad / deepfake
The detector head is **3-class** (`real`=0, `pad`=1, `deepfake`=2), trained with CrossEntropy.
Ground-truth labels are resolved per image by `gsd.get_label.get_label_all` (the authoritative
PAAS labeler), **not** the json `cls_label` (which is binary and disagrees with path on pad images).
**MAKEUP attacks are folded into PAD**; `UNKNOWN` images are dropped. Class imbalance is handled with
inverse-frequency CrossEntropy weights (`class_weight=true`). The operational **real-vs-fake AUC**
(`P(fake)=1-P(real)`) is the default `best.pt` selection metric (`select_metric=bin_auc`); set it to
`acc` or `bal_acc` to optimise overall / balanced 3-class accuracy instead. Set `num_classes=2` to
fall back to a pure real/fake head.

## Data
Defaults point at the MIDS datasets (each a json list of `{"image", "cls_label", ...}`):
```
train: /datasets/work/vLLM/temp/testset/testset_mids/mids_first_half.json   (~1.71M imgs)
val  : /datasets/work/vLLM/temp/testset/testset_mids/mids_testset.json      (~30k imgs)
```
`eval_limit` caps val images used *during* training (for speed); run `eval.py` for the full set.

## Method (as implemented)

Dual-stream, per the paper:

1. **Frozen stream** — CLIP ViT-L/14-336 (always frozen) produces a global *guide* vector per image
   `g_i ∈ ℝ^D` (Global-Avg-Pool over patch tokens; the paper's "semantic consensus").
2. **Semantic basis (per batch, Householder QR):**
   ```
   c = (1/B) Σ g_i                         # semantic anchor (batch centroid)
   G = [g_1-c, …, g_B-c] ∈ ℝ^{D×B}         # centered guide matrix
   G = Q R   (Householder QR)              # torch.linalg.qr
   U = Q[:, :K] ∈ ℝ^{D×K}                  # K=16 semantic directions
   ```
   `U` comes from the **frozen** guides → detached, recomputed every batch (no running stats).
3. **Trainable stream** — a CLIP ViT-L/14 copy whose **final 4 encoder layers** have GSD injected:
   each such layer's patch tokens are projected to the orthogonal complement
   ```
   F'_l = F_l (I − U Uᵀ)
   ```
   (the [CLS] token is left intact). The de-semanticized final features are pooled → a linear head → 1 logit.
4. **Loss** — CrossEntropy over the 3 classes (inverse-frequency weighted); **AdamW** (backbone lr
   `1e-6`, head lr `1e-4`); augment with horizontal flip + Gaussian blur + JPEG recompression.
   (The paper is binary BCE; the 3-class softmax head is the project-specific extension.)

Defaults match the paper's ablation winners: `K=16`, `n_gsd_layers=4`, guide = Global-Avg-Pool,
Householder QR.

### Train vs. inference
`U` is estimated from the **current batch's** frozen guides at *both* train and test — there are no
running statistics, so nothing is "updated" at test time. GSD therefore needs **batch_size ≥ 2**.
For single-image inference, freeze a stable basis from reference images via `infer.py --anchor-dir`.

## Layout
```
base_models/clip-vit-large-patch14-336   vendored CLIP vision weights (standalone)
gsd/get_label.py    vendored authoritative label resolver (get_label_all)
gsd/householder.py  semantic-subspace basis (Householder QR, + SVD option)
gsd/projection.py   F'(I − UUᵀ) de-semanticization
gsd/model.py        dual-stream CLIP + GSD forward-hooks on the last N layers + num_classes head
gsd/data.py         3-class dataset (label via get_label_all; MAKEUP→PAD) + blur/JPEG aug
gsd/engine.py       CrossEntropy train loop + 3-class + real-vs-fake AUC eval
train.py eval.py infer.py   CLIs
scripts/verify_gsd.py       math + forward/grad self-test (no data needed)
configs/default.json        default config (3-class, MIDS datasets wired in)
```

## Usage
```bash
# 0) sanity-check the implementation (no data)
python3.12 scripts/verify_gsd.py

# 1) train — datasets + 3-class are the config defaults, so this is all you need:
python3.12 train.py --config configs/default.json
#    (override anything, e.g. a smaller batch:)
python3.12 train.py --config configs/default.json --set batch_size=64 output_dir=runs/gsd_b64

# 2) evaluate on the full test json
python3.12 eval.py --ckpt runs/gsd/best.pt --data /datasets/work/vLLM/temp/testset/testset_mids/mids_testset.json

# 3) infer (batched; folder of query images) -> per-class probs + fake_prob
python3.12 infer.py --ckpt runs/gsd/best.pt --input /path/imgs --out preds.jsonl
```

### Single-image inference (embedded anchor)
GSD's semantic basis `U` is normally estimated per batch, so a lone image has no batch centroid.
`train.py` solves this by building a **fixed anchor `U` from the testset** (`anchor_data` or `val_data`,
`anchor_limit` strided refs) and **embedding it in every checkpoint** (`anchor_U`). So single-image
scoring works out of the box — `eval.load_model` installs the embedded anchor automatically:
```bash
python3.12 infer.py --ckpt runs/gsd/best.pt --input one.jpg     # uses the embedded anchor
```
Overrides (optional): `--anchor anchor.pt` (a standalone basis from `build_anchor.py`) or
`--anchor-dir /imgs` (recompute on the fly). If a checkpoint has *no* embedded anchor and none is
passed, a single image runs with **GSD disabled** (warned) — batches of ≥2 images estimate `U`
themselves and need no anchor.

Key knobs (`--set num_classes=2|3 select_metric=bin_auc|acc|bal_acc class_weight=true|false
k=… n_gsd_layers=… guide_pool=gap|cls qr_method=householder|svd trainable=full|lastN|head`).

**CPU cap (training):** `cpu_fraction` limits OMP/MKL/BLAS + torch threads to that fraction of the
available cores (default `0.5` = 50%; set before torch/OMP init, and DataLoader workers inherit it).
Override per-run with `--set cpu_fraction=0.25` or the `GSD_CPU_FRACTION` env var.

**GPUs:** training runs on the GPU list `gpus` (default `"0,1,2,3"`, via `nn.DataParallel`). Edit it
freely — `"0"` for single-GPU, `"0,1"` for two, etc. The global batch is split across the listed
GPUs, so keep `batch_size / #gpus ≥ 2` (GSD needs ≥2 per GPU to estimate `U`). Eval runs on the
primary GPU. (You can also restrict via `CUDA_VISIBLE_DEVICES`, but editing `gpus` is the intended knob.)

**Log file:** every run tees stdout/stderr to `<output_dir>/train.log` (timestamped, appended).
`trainable=head` (freeze the trainable backbone, train only the head on de-semanticized frozen
features) is the cheapest way to validate GSD on limited compute.
