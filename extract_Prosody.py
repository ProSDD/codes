import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from tqdm import tqdm
import torch
import torch.nn.functional as F

from masked_prosody_model import MaskedProsodyModel

SAMPLING_RATE = 16000
DUR_SEC = 4.0
TARGET_SAMPLES = int(SAMPLING_RATE * DUR_SEC)  # 64000
TARGET_FRAMES = 200  # 4s at 20ms


def ffmpeg_load(audio_path, target_sr=SAMPLING_RATE):
    command = [
        "ffmpeg", "-y", "-i", audio_path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(target_sr),
        "-hide_banner", "-loglevel", "error", "-"
    ]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()}")
    return np.frombuffer(proc.stdout, dtype=np.float32), target_sr

def load_audio_16k(path: str) -> np.ndarray:
    try:
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32, copy=False)
        if sr != SAMPLING_RATE:
            wav, _ = ffmpeg_load(path, target_sr=SAMPLING_RATE)
        return wav
    except Exception:
        wav, _ = ffmpeg_load(path, target_sr=SAMPLING_RATE)
        return wav

def pad(wav: torch.Tensor, max_len: int = TARGET_SAMPLES) -> torch.Tensor:
    n = wav.numel()
    if n >= max_len:
        return wav[:max_len]
    num_repeats = (max_len // n) + 1
    padded_wav = wav.repeat(num_repeats)
    return padded_wav[:max_len]

def padm(wav, max_len: int = TARGET_SAMPLES) -> torch.Tensor:
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav)
    n = wav.numel()
    if n >= max_len:
        s = (n - max_len) // 2
        return wav[s:s + max_len]
    pad_total = max_len - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return F.pad(wav, (pad_left, pad_right))


@torch.inference_mode()
def mpm_rep_200(model, wav4: np.ndarray, layer: int = 7, device: torch.device = torch.device("cpu")) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, wav4, SAMPLING_RATE)
        try:
            rep = model.process_audio(tmp.name, layer=layer, device="cpu")
        except TypeError:
            rep = model.process_audio(tmp.name, layer=layer)

    if isinstance(rep, torch.Tensor):
        rep_t = rep
    else:
        rep_t = torch.as_tensor(rep)
    rep_t = rep_t.cpu()

    T, D = rep_t.shape
    if T >= 2 * TARGET_FRAMES:
        rep_t = rep_t[: 2 * TARGET_FRAMES]                 # (400, D)
        rep_t = rep_t.reshape(TARGET_FRAMES, 2, D).mean(1) # (200, D)
        return rep_t.detach().cpu().numpy()
    x = rep_t.T.unsqueeze(0)  # (1, D, T)
    y = F.interpolate(x, size=TARGET_FRAMES, mode="linear", align_corners=False)
    rep200 = y.squeeze(0).T   # (200, D)
    return rep200.detach().cpu().numpy()

def main(protocol_txt: str, audio_dir: str, out_txt: str, utt_col: int = 1, ext: str = ".flac", layer: int = 7,
         device_str: str = "cuda"):
    device = torch.device(device_str if (device_str.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print("Using device:", device, flush=True)
    model = MaskedProsodyModel.from_pretrained("cdminix/masked_prosody_model")
    model = model.to("cpu")
    model.eval()
    try:
        print("Model param device:", next(model.parameters()).device, flush=True)
    except StopIteration:
        print("WARN: model has no parameters? (unexpected)", flush=True)
    with open(protocol_txt, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    with open(out_txt, "w") as fout:
        for ln in tqdm(lines, desc="Extracting MPM"):
            parts = ln.split()
            if len(parts) <= utt_col:
                continue
            utt = parts[utt_col]
            audio_path = os.path.join(audio_dir, utt)
            if not os.path.exists(audio_path):
                print(f"[WARN] Skip {utt}: Not found.", flush=True)
                continue
            try:
                wav_np = load_audio_16k(audio_path)
                wav4_tensor = padm(wav_np, TARGET_SAMPLES)
                wav4_np = wav4_tensor.numpy()
                rep200 = mpm_rep_200(model, wav4_np, layer=layer, device=device)  # (200, D)
                frames = [",".join(f"{v:.6f}" for v in row) for row in rep200]
                fout.write(f"{utt}\t{'|'.join(frames)}\n")
            except Exception as e:
                print(f"[ERROR] {utt}: {str(e)}", flush=True)
                
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_txt", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--utt_col", type=int, default=0)
    ap.add_argument("--ext", default=".flac")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--device", default="cuda") 
    args = ap.parse_args()
    main(
        protocol_txt=args.protocol_txt,
        audio_dir=args.audio_dir,
        out_txt=args.out_txt,
        utt_col=args.utt_col,
        ext=args.ext,
        layer=args.layer,
        device_str=args.device,
    )