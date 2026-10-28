"""Train the official UNetLidar DDPM and sample synthetic BCI IV-2a trials.

Targets the RTX 3060 Ti (8 GB). This is the faithful Stage-1 diffusion path the
official DDF-Net README describes, wired to our ``.gdf`` instead of the BNCI
``.mat`` files:

  1. load real 25-channel trials and interpolate every channel to ``166 x 600``;
  2. probe the largest feasible batch size (``8 -> 4 -> 2``);
  3. train ``UNetLidar`` to predict DDPM noise (``epochs=100``, tqdm, loss + GPU
     memory printed every 10 epochs);
  4. DDPM-sample ``--n-gen`` synthetic trials, inverse-interpolate and merge the
     25 channels back to 1-D;
  5. write ``merged/{1..N}.mat`` (field ``merged_data``) + ``labels.npy`` in the
     exact format ``convert_to_npy.py`` expects.

Run:
  py train_unetlidar.py --subject A01 --trials 40 --epochs 100 \
      --batch-size 8 --n-gen 144 --out-dir stage1_out
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent                       # Diffusion/diffusion
_DDF = _HERE.parents[1]                                       # adapters/ddf_local
for p in (str(_HERE), str(_DDF)):
    if p not in sys.path:
        sys.path.insert(0, p)

from diffusion import Diffusion                                # noqa: E402
from unetIDDPM import UNetLidar                                # noqa: E402
from utils import AvgrageMeter                                  # noqa: E402
from run_stage1_pipeline import (                               # noqa: E402
    load_25ch_trials, interpolate, restore, TARGET_SHAPE, N_CHANNELS)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[warn] CUDA not available; falling back to CPU (training will be slow)")
    return torch.device("cpu")


def build_training_images(data):
    """Interpolate every channel of every trial to ``[N, 1, 166, 600]``."""
    n_trials, n_ch, n_times = data.shape
    images = []
    x_old = x_new = None
    for t in range(n_trials):
        for c in range(n_ch):
            interp, x_old, x_new = interpolate(data[t, c])
            images.append(interp)
    imgs = np.stack(images).astype(np.float32)[:, None]        # [N, 1, 166, 600]
    return imgs, x_old, x_new


def probe_batch_size(model, diffusion, imgs, device, steps=(8, 4, 2)):
    """Largest batch size that fits without spilling past VRAM: try 8 -> 4 -> 2.

    CUDA silently overcommits into shared system RAM instead of raising OOM, so a
    batch can "succeed" yet crawl ~25x slower. We reject any batch whose peak
    allocation exceeds 85% of the device's total memory, not just hard OOM.
    """
    model.train()
    total = torch.cuda.get_device_properties(device).total_memory
    limit = int(0.85 * total)
    for bs in steps:
        bs = min(bs, len(imgs))
        try:
            torch.cuda.reset_peak_memory_stats()
            x = torch.from_numpy(imgs[:bs]).to(device)
            t = torch.randint(0, diffusion.T, (bs,), device=device)
            x_noisy, noise = diffusion.forward_diffusion_sample(x, t, device)
            noise_pred = model(x_noisy, t)
            F.l1_loss(noise, noise_pred).backward()
            model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated()
            torch.cuda.empty_cache()
            if peak > limit:
                print(f"  [probe] batch_size={bs} spills ({peak / 2**30:.2f} GiB "
                      f"> {limit / 2**30:.2f} GiB limit), trying smaller")
                continue
            print(f"  [probe] batch_size={bs} fits (peak {peak / 2**30:.2f} GiB)")
            return bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [probe] OOM at batch_size={bs}, trying smaller")
    raise RuntimeError("no feasible batch size (tried 8, 4, 2)")


def train(model, diffusion, imgs, device, epochs, batch_size, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(imgs)
    rng = np.random.default_rng(0)
    last_loss = float("nan")

    pbar = tqdm(range(1, epochs + 1), desc="train", unit="epoch")
    for epoch in pbar:
        model.train()
        idx = rng.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            x = torch.from_numpy(imgs[idx[i:i + batch_size]]).to(device)
            t = torch.randint(0, diffusion.T, (x.shape[0],), device=device)
            x_noisy, noise = diffusion.forward_diffusion_sample(x, t, device)
            noise_pred = model(x_noisy, t)
            loss = F.l1_loss(noise, noise_pred)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if epoch == 1 or epoch % 10 == 0:
            mem = torch.cuda.memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
            msg = f"loss={last_loss:.6f} mem={mem:.2f}GB"
            pbar.set_postfix_str(msg)
            print(f"[epoch {epoch}/{epochs}] {msg}")

    return last_loss


def sample_trials(model, diffusion, x_old, x_new, n_trials, n_ch, device,
                  out_dir):
    """DDPM-reverse sample n_trials x n_ch images, restore, merge, save .mat."""
    merged_dir = Path(out_dir) / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    n_times = len(x_old)

    merged = np.empty((n_trials, n_ch, n_times), dtype=np.float64)
    for tr in tqdm(range(n_trials), desc="sample", unit="trial"):
        # one trial = n_ch channels, denoised together as a [n_ch, 1, 166, 600] batch
        img = torch.randn((n_ch, 1, *TARGET_SHAPE), device=device)
        with torch.no_grad():
            for i in reversed(range(diffusion.T)):
                t = torch.full((1,), i, device=device, dtype=torch.long)
                img = diffusion.sample_timestep(img, t, model)
        arr = img.squeeze(1).cpu().numpy()                    # [n_ch, 166, 600]
        for c in range(n_ch):
            merged[tr, c] = restore(arr[c], x_old, x_new)

    # convert_to_npy.py reads .mat with field `merged_data` [n_times x 25]
    from scipy.io import savemat
    for tr in range(n_trials):
        savemat(merged_dir / f"{tr + 1}.mat", {"merged_data": merged[tr].T})
    return merged


def main(args):
    device = get_device()
    print(f"[device] {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    print(f"[load] {args.subject}{args.run} -> {args.trials} balanced trials")
    data, train_labels = load_25ch_trials(args.subject, args.run, args.raw_dir,
                                          args.trials, args.seed)
    print(f"  trials={data.shape[0]} channels={data.shape[1]} time={data.shape[2]}")

    imgs, x_old, x_new = build_training_images(data)
    print(f"[interp] {imgs.shape[0]} training images of {imgs.shape[2:]} (in_ch=1)")

    diffusion = Diffusion(T=args.t)
    model = UNetLidar(T=args.t, in_ch=1, out_ch=1, dropout=args.dropout,
                      tdim=args.tdim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] UNetLidar params={n_params / 1e6:.2f}M  T={args.t}")

    if device.type == "cuda":
        batch_size = probe_batch_size(model, diffusion, imgs, device,
                                      steps=(args.batch_size, 4, 2))
    else:
        batch_size = args.batch_size
    print(f"[batch] largest feasible batch_size = {batch_size}")

    t0 = time.time()
    final_loss = train(model, diffusion, imgs, device, args.epochs, batch_size,
                       args.lr)
    print(f"[train] done in {time.time() - t0:.1f}s, final loss={final_loss:.6f}")

    ckpt = Path(args.out_dir) / "unetlidar.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print(f"[ckpt] saved {ckpt}")

    merged = sample_trials(model, diffusion, x_old, x_new, args.n_gen,
                           data.shape[1], device, args.out_dir)

    classes = np.unique(train_labels)
    per = args.n_gen // len(classes)
    labels = np.empty(args.n_gen, dtype=int)
    for k, c in enumerate(classes):
        labels[k * per:(k + 1) * per] = c
    labels[len(classes) * per:] = classes[0]  # remainder, if n_gen not divisible
    np.save(Path(args.out_dir) / "labels.npy", labels)
    print(f"[out] merged/{{1..{args.n_gen}}}.mat  ({args.n_gen} x "
          f"{merged.shape[1]} x {merged.shape[2]})")
    print(f"[out] labels.npy  {dict(zip(*np.unique(labels, return_counts=True)))}")
    print("Done. Next: py convert_to_npy.py --merged-dir <out>/merged "
          "--labels <out>/labels.npy --out <out>/ddf_synthetic")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="A01")
    ap.add_argument("--run", default="T")
    ap.add_argument("--raw-dir", default=str(_HERE.parents[3] / "data" / "raw"))
    ap.add_argument("--out-dir", default=str(_DDF / "stage1_out"))
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-gen", type=int, default=144)
    ap.add_argument("--t", type=int, default=100, help="DDPM timesteps")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tdim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args)
