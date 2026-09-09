#!/usr/bin/env python3.12
"""PAAS_v4 (GSD) batch tester over a folder tree of images + videos -- multi-GPU batching.

Mirrors PAAS_ensemble_v2/test_video_image_batch.py (same discovery, file-sharding across GPUs,
real cross-file batching, and the SAME unified line format) but drives a single GSD detector:

  * recurses --input-dir, finds every still image + video, FILE-SHARDS them round-robin across the
    selected GPUs (one spawn worker per GPU, each pinned to its physical device == cuda:0);
  * REAL batching: each worker accumulates frames/images ACROSS files and scores them in one forward
    per --flush-size (GSD estimates its semantic subspace per batch; the checkpoint's embedded
    anchor U is also loaded so scoring is deterministic regardless of batch composition);
  * ground truth = the `real`/`fake` path component;
  * pred = fake if P(fake) >= threshold else real; `type` = argmax 3-class name (real/pad/deepfake);
    P(fake) = 1 - P(real);
  * copies every misclassified item to --miss-dir;
  * each worker writes shard-suffixed results; the main process merges them into results_gsd.txt in
    the unified format  (OK/XX/SK/ER  truth=..  pred=..  type=..  fake=..  match=..  <path>)
    and prints a per-label summary.

Examples:
  python3.12 test_video_image_batch.py --ckpt runs/gsd/best.pt \
      --input-dir /datasets/work/vLLM/data/axonlabs_data_1 --out-dir runs/test --devices all
  python3.12 test_video_image_batch.py --ckpt runs/gsd/best.pt --input-dir /data --device 0
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

PROJECT = "PAAS_v4"
_HERE = os.path.dirname(os.path.abspath(__file__))

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
VID_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg")


# ----------------------------------------------------------------------------- unified line format
def fmt_line(tag, truth, pred, ftype, fake, match, path):
    """Identical to PAAS_ensemble_v2/paas/io_results.fmt_line."""
    fs = "------" if fake is None else f"{float(fake):.4f}"
    ms = "------" if match is None else f"{float(match):.4f}"
    tf = f"type={ftype}"
    tf = tf + " " * max(2, 15 - len(tf))
    return f"{tag}  truth={truth}  pred={pred}  {tf}fake={fs} match={ms}  {path}"


# ----------------------------------------------------------------------------- discovery / paths
def truth_of(path):
    parts = path.lower().split(os.sep)
    if "real" in parts:
        return "real"
    if "fake" in parts:
        return "fake"
    return None


def iter_media(root, skip_dirs=()):
    skip_dirs = tuple(os.path.abspath(d) for d in skip_dirs if d)
    for dp, _, files in os.walk(root):
        adp = os.path.abspath(dp)
        if any(adp == s or adp.startswith(s + os.sep) for s in skip_dirs):
            continue
        for fn in sorted(files):
            ext = os.path.splitext(fn)[1].lower()
            kind = "image" if ext in IMG_EXT else ("video" if ext in VID_EXT else None)
            if kind is not None:
                yield os.path.join(dp, fn), kind


def frames_of(path, kind, stride):
    import cv2
    if kind == "image":
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is not None:
            yield path, cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            yield f"{path}#frame={i:06d}", cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()


def miss_target(miss_dir, truth, key):
    flat = key.replace("#frame=", "_frame").lstrip("/").replace("/", "__")
    return os.path.join(miss_dir, truth, flat)


# ----------------------------------------------------------------------------- device selection
def parse_devices(args):
    import torch
    n = torch.cuda.device_count()
    if args.device is not None:
        raw = str(args.device).strip().lower().replace("cuda:", "")
        return [int(raw)], n
    spec = args.devices.strip().lower()
    if spec == "all":
        return list(range(n)), n
    return [int(x.replace("cuda:", "")) for x in spec.split(",") if x.strip()], n


# ----------------------------------------------------------------------------- GPU worker
def gpu_worker(device_id, worker_idx, media_paths, args_dict, n_workers, resume_files, result_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    args = argparse.Namespace(**args_dict)

    import cv2
    import torch
    from PIL import Image
    sys.path.insert(0, _HERE)
    from gsd.data import build_transform
    from gsd.config import CLASS_NAMES
    from eval import load_model

    try:
        frac = float(getattr(args, "cpu_frac", 0.9))
        per = max(2, int((os.cpu_count() or 8) * frac) // max(1, n_workers))
        torch.set_num_threads(per)
        cv2.setNumThreads(per)   # multithreaded video decode / colour-convert (the CPU bottleneck)
        print(f"[GPU {device_id}/w{worker_idx}] CPU threads/worker={per} (cpu_frac={frac})", flush=True)
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    device = "cuda:0"
    model, cfg = load_model(args.ckpt, device)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
        getattr(cfg, "amp_dtype", "bf16"), torch.bfloat16)
    num_classes = cfg.num_classes
    names = list(CLASS_NAMES) if num_classes == 3 else ["real", "fake"]
    tfm = build_transform(cfg.image_size, train=False, cfg=cfg)
    thr = args.threshold if args.threshold is not None else 0.5
    embedded = getattr(model, "_fixed_U", None) is not None
    print(f"[GPU {device_id}] GSD ready (num_classes={num_classes}, embedded_anchor={embedded}) "
          f"threshold={thr:.4f} ({len(media_paths)} media files in shard)", flush=True)

    sfx = f".shard{worker_idx}"
    shard_path = os.path.join(args.out_dir, f"results_gsd{sfx}.txt")

    # result-line = "<status> truth=.. pred=.. type=.. fake=.. match=.. <path>"; the path (last
    # field) may itself contain spaces, so peel off exactly the 6 fixed fields and keep the rest.
    import re as _re

    def _key_of(line):
        p = _re.split(r"\s+", line.rstrip("\n").strip(), maxsplit=6)
        return p[6] if len(p) == 7 and p[6].startswith("/") else ""

    # -- precomputed SK list: skip exactly the reals the ensemble skipped (no insightface) --
    sk_set = set()
    sk_path = getattr(args, "sk_list", "") or ""
    if sk_path and os.path.isfile(sk_path):
        for ln in open(sk_path):
            if ln.startswith("SK"):
                k = _key_of(ln)
                if k:
                    sk_set.add(k)
        print(f"[GPU {device_id}] loaded {len(sk_set)} SK real-keys from {os.path.basename(sk_path)} "
              f"(insightface OFF)", flush=True)

    # -- resume: re-use frames scored in prior run(s), read from ALL pre-existing shard files so it
    #    is independent of how files are re-sharded across workers this run. A file is skipped
    #    (no decode) only if it was FULLY processed before; the last file each prior shard touched
    #    may be partial, so those are re-scanned frame-by-frame (already-scored frames are skipped). --
    done_keys, done_files, incomplete_files = set(), set(), set()
    resume = bool(getattr(args, "resume", 0)) and bool(resume_files)
    if resume:
        for rf in resume_files:
            if not os.path.isfile(rf):
                continue
            rf_last = None
            for ln in open(rf):
                if not ln or ln.startswith("#"):
                    continue
                k = _key_of(ln)
                if k:
                    done_keys.add(k)
                    fpart = k.split("#frame=")[0]
                    done_files.add(fpart)
                    rf_last = fpart
            if rf_last is not None:
                incomplete_files.add(rf_last)     # possibly-partial file for that shard
        print(f"[GPU {device_id}/w{worker_idx}] resume: {len(done_keys)} keys / {len(done_files)} files "
              f"done ({len(incomplete_files)} possibly-partial) from {len(resume_files)} prior shard(s)",
              flush=True)
    f_out = open(shard_path, "w")

    rfilter = None
    if getattr(args, "filter_real", 0) and not sk_set:
        try:
            sys.path.insert(0, "/datasets/work/vLLM/temp/PAAS_ensemble_v2")
            from paas.data.face_filter import FaceQualityFilter
            rfilter = FaceQualityFilter()
            print(f"[GPU {device_id}] face-quality filter ON for reals (matches ensemble)", flush=True)
        except Exception as e:
            print(f"[GPU {device_id}] --filter-real requested but filter unavailable ({e}); reals NOT filtered.",
                  flush=True)

    tally = {}
    counts = {"frames": 0, "skipped": 0, "errors": 0, "miss_saved": 0}
    buffer = []
    t0 = time.time()
    last_report = [time.time()]

    def handle(it, p_fake, pred_cls):
        key, truth, rgb = it["key"], it["truth"], it["rgb"]
        pred = "fake" if p_fake >= thr else "real"
        ftype = names[pred_cls]
        correct = (pred == truth)
        f_out.write(fmt_line("OK" if correct else "XX", truth, pred, ftype, p_fake, None, key) + "\n")
        t = tally.setdefault(truth, [0, 0]); t[0] += correct; t[1] += 1
        if args.copy_miss and not correct:
            dst = miss_target(args.miss_dir, truth, key)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                cv2.imwrite(dst, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                counts["miss_saved"] += 1
            except Exception:
                pass

    def flush():
        if not buffer:
            return
        try:
            x = torch.stack([tfm(Image.fromarray(it["rgb"])) for it in buffer]).to(device)
            with torch.autocast(device_type="cuda", dtype=amp, enabled=(amp != torch.float32)):
                logits = model(x)
            probs = torch.softmax(logits.float(), dim=1)
            p_fake = (1.0 - probs[:, 0]).cpu().numpy()
            pred_cls = probs.argmax(dim=1).cpu().numpy()
            for it, pf, pc in zip(buffer, p_fake, pred_cls):
                handle(it, float(pf), int(pc))
                it["rgb"] = None
        except Exception as exc:
            for it in buffer:
                counts["errors"] += 1
                f_out.write(fmt_line("ER", it["truth"], "----", "error", None, None, it["key"]) + "\n")
            print(f"[GPU {device_id}] batch-error {exc!r}", file=sys.stderr, flush=True)
        counts["frames"] += len(buffer)
        buffer.clear()
        if args.progress_interval > 0 and time.time() - last_report[0] >= args.progress_interval:
            rate = counts["frames"] / max(time.time() - t0, 1e-9)
            done = sum(v[1] for v in tally.values())
            print(f"[GPU {device_id}] {counts['frames']} frames  {rate:.0f}/s  eval={done} "
                  f"err={counts['errors']}", flush=True)
            last_report[0] = time.time()

    try:
        for path, kind in media_paths:
            truth = truth_of(path)
            if truth is None:
                continue
            # resume: fully-processed files are skipped without re-decoding; possibly-partial
            # files are re-scanned (their already-scored frames are dropped per-frame below)
            if resume and path in done_files and path not in incomplete_files:
                continue
            for key, rgb in frames_of(path, kind, args.frame_stride):
                if resume and key in done_keys:
                    continue
                if truth == "real":
                    skip = (key in sk_set) if sk_set else \
                           (rfilter is not None and not rfilter.passes(rgb))
                    if skip:
                        counts["skipped"] += 1
                        f_out.write(fmt_line("SK", truth, "skip", "lowqual", None, None, key) + "\n")
                        continue
                buffer.append({"key": key, "truth": truth, "rgb": rgb})
                if len(buffer) >= args.flush_size:
                    flush()
        flush()
    except Exception as exc:
        print(f"[GPU {device_id}] WORKER-ERROR {exc!r}", file=sys.stderr, flush=True)
    finally:
        f_out.close()
        result_q.put({"device_id": device_id, "tally": tally, "counts": counts})


# ----------------------------------------------------------------------------- merge / summary
def merge_shard_files(out_dir, basename, device_ids, header_lines):
    import glob as _glob
    import re as _re
    shards = sorted(_glob.glob(os.path.join(out_dir, f"results_{basename}.shard*.txt")))
    if not shards:
        return None

    def _key(line):                       # path (last field) may contain spaces
        p = _re.split(r"\s+", line.rstrip("\n").strip(), maxsplit=6)
        return p[6] if len(p) == 7 and p[6].startswith("/") else None

    # dedup by frame-key (scores are deterministic; keep first seen); drops prior-run duplicates
    seen, kept = set(), []
    for s in shards:
        for ln in open(s):
            k = _key(ln)
            if k is None or k in seen:
                continue
            seen.add(k)
            kept.append(ln if ln.endswith("\n") else ln + "\n")
    dst = os.path.join(out_dir, f"results_{basename}.txt")
    with open(dst, "w") as out:
        for h in header_lines:
            out.write(h + "\n")
        out.writelines(kept)
    for s in shards:
        try:
            os.remove(s)
        except OSError:
            pass
    return dst


def main():
    ap = argparse.ArgumentParser(description=f"{PROJECT} batch image/video tester (multi-GPU)")
    ap.add_argument("--ckpt", default="runs/gsd/best.pt")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", default="runs/test")
    ap.add_argument("--miss-dir", default=None, help="copy misses here (default: <out-dir>/miss)")
    ap.add_argument("--devices", default="all", help="CUDA devices: 'all' or e.g. '0,1,2,3'.")
    ap.add_argument("--device", default=None, help="single-GPU alias (int or cuda:N); overrides --devices.")
    ap.add_argument("--threshold", type=float, default=None, help="fake-decision threshold (default 0.5)")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="cap total media files (0 = all)")
    ap.add_argument("--flush-size", type=int, default=256,
                    help="frames accumulated ACROSS files per forward (the real-batch knob)")
    ap.add_argument("--copy-miss", type=int, default=1, help="1 = copy misses to --miss-dir")
    ap.add_argument("--filter-real", type=int, default=0,
                    help="1 = skip low-quality reals via FaceQualityFilter (SK, excluded) - matches ensemble")
    ap.add_argument("--sk-list", default="",
                    help="path to a results file whose 'SK' lines give real frame-keys to skip; "
                         "when set, reals are filtered by this precomputed list (no insightface) "
                         "so the evaluated real set is byte-identical to the ensemble run")
    ap.add_argument("--resume", type=int, default=0,
                    help="1 = append to existing shard files, re-using frames already scored "
                         "(skip fully-processed files and already-written frame keys)")
    ap.add_argument("--cpu-frac", type=float, default=0.9,
                    help="fraction of host CPUs used per model (split across its GPU workers) for "
                         "torch + OpenCV decode threads")
    ap.add_argument("--workers-per-gpu", type=int, default=1,
                    help="decode/score worker processes per GPU; >1 runs several videos concurrently "
                         "per GPU to beat the serial-decode bottleneck (GPU is far from saturated)")
    ap.add_argument("--progress-interval", type=float, default=10.0)
    args = ap.parse_args()

    if not os.path.isabs(args.ckpt):
        args.ckpt = os.path.join(_HERE, args.ckpt)

    import torch
    if not torch.cuda.is_available():
        print("CUDA is required for multi-GPU batch testing.", file=sys.stderr)
        return 1
    if not os.path.isfile(args.ckpt):
        print(f"checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 1
    devices, n_visible = parse_devices(args)
    bad = [d for d in devices if d < 0 or d >= n_visible]
    if not devices or bad:
        print(f"Invalid devices {bad or devices}; visible CUDA count = {n_visible}.", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    args.miss_dir = args.miss_dir or os.path.join(args.out_dir, "miss")

    media = list(iter_media(args.input_dir, skip_dirs=(args.miss_dir, args.out_dir)))
    media = [(p, k) for p, k in media if truth_of(p) is not None]
    if args.limit:
        media = media[:args.limit]
    if not media:
        print(f"No labelled image/video files under {args.input_dir}")
        return 0
    n_images = sum(1 for _, k in media if k == "image")
    n_videos = sum(1 for _, k in media if k == "video")
    print(f"[{PROJECT}] media: {n_images} images + {n_videos} videos = {len(media)} files | "
          f"ckpt={args.ckpt} | GPUs={','.join(map(str, devices))}", flush=True)

    import glob
    wpg = max(1, int(getattr(args, "workers_per_gpu", 1)))
    worker_devs = [d for d in devices for _ in range(wpg)]      # e.g. [0,0,1,1] for wpg=2
    n_workers = len(worker_devs)

    # pre-existing shard files are the resume source; new workers get non-colliding suffixes so
    # they never clobber the prior run's shards (which the merge still globs at the end).
    resume_files = sorted(glob.glob(os.path.join(args.out_dir, "results_gsd.shard*.txt"))) \
        if getattr(args, "resume", 0) else []

    def _shard_num(p):
        try:
            return int(os.path.basename(p).split(".shard")[-1].split(".txt")[0])
        except Exception:
            return -1
    base_idx = max((_shard_num(p) for p in resume_files), default=-1) + 1   # never reuse a prior suffix

    shards = [media[i::n_workers] for i in range(n_workers)]
    print(f"[{PROJECT}] {n_workers} workers ({wpg}/gpu) over GPUs {devices}; "
          f"resume from {len(resume_files)} prior shard(s)", flush=True)
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    procs = []
    for j, (dev, shard) in enumerate(zip(worker_devs, shards)):
        widx = base_idx + j
        p = ctx.Process(target=gpu_worker,
                        args=(dev, widx, shard, vars(args), n_workers, resume_files, result_q))
        p.start()
        procs.append(p)

    tally = {}
    counts = {"frames": 0, "skipped": 0, "errors": 0, "miss_saved": 0}
    for _ in procs:
        msg = result_q.get()
        for truth, (c, n) in msg["tally"].items():
            t = tally.setdefault(truth, [0, 0]); t[0] += c; t[1] += n
        for k, v in msg["counts"].items():
            counts[k] = counts.get(k, 0) + v
        print(f"[GPU {msg['device_id']}-DONE] frames={msg['counts']['frames']} "
              f"err={msg['counts']['errors']}", flush=True)
    for p in procs:
        p.join()

    col = "# columns: OK/XX/SK/ER  truth  pred  type  fake_score  match_score  image"
    tag = f"# {PROJECT} | ckpt={args.ckpt} | input={args.input_dir}"
    out_file = merge_shard_files(args.out_dir, "gsd", devices, [tag, col])

    def pct(c, n):
        return 100.0 * c / n if n else float("nan")
    print(f"\n=== {PROJECT} accuracy (threshold-based; SK excluded) ===")
    recalls = []
    for truth, (c, n) in sorted(tally.items()):
        recalls.append(pct(c, n))
        print(f"  {truth:5s}: {c}/{n} = {pct(c, n):.2f}%")
    tot_c = sum(c for c, _ in tally.values())
    tot_n = sum(n for _, n in tally.values())
    print(f"  OVERALL: {tot_c}/{tot_n} = {pct(tot_c, tot_n):.2f}%"
          + (f"   (balanced = {sum(recalls)/len(recalls):.2f}%)" if recalls else ""))
    print(f"  frames={counts['frames']} errors={counts['errors']} miss-saved={counts['miss_saved']}")
    print(f"\nresults -> {out_file}  | misses -> {args.miss_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
