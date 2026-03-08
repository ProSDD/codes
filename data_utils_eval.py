import os
import subprocess
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

SAMPLING_RATE = 16000
TARGET_SAMPLES = 64000  # 4.0s

def ffmpeg_load(audio_path, target_sr=SAMPLING_RATE):
    """Robust ffmpeg loading"""
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(target_sr),
        "-hide_banner", "-loglevel", "error", "-"
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        audio = np.frombuffer(proc.stdout, dtype=np.float32)
        return torch.from_numpy(audio)
    except Exception:
        return torch.zeros(TARGET_SAMPLES)
    
def padr(wav: torch.Tensor, max_len: int = TARGET_SAMPLES) -> torch.Tensor:
    n = wav.numel()
    if n >= max_len:
        return wav[:max_len]
    num_repeats = (max_len // n) + 1
    padded_wav = wav.repeat(num_repeats)
    return padded_wav[:max_len]

def pad(wav: torch.Tensor, max_len: int = TARGET_SAMPLES) -> torch.Tensor:
    n = wav.numel()
    if n >= max_len:
        s = (n - max_len) // 2
        return wav[s:s + max_len]
    pad_total = max_len - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return F.pad(wav, (pad_left, pad_right))

class ProSDDEvalDataset(Dataset):
    def __init__(self, list_path, wav_dir, ext="flac"):
        self.wav_dir = wav_dir
        self.ext = ext.lstrip(".")
        self.utt_ids = []
        with open(list_path) as f:
            for line in f:
                if not line.strip(): continue
                parts = line.strip().split()
                self.utt_ids.append(parts[1]) 

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        path = os.path.join(self.wav_dir, f"{utt_id}.{self.ext}")
        #path = os.path.join(self.wav_dir, f"{utt_id}")
        try:
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1: wav = wav.mean(dim=0)
            wav = wav.squeeze()
            if sr != SAMPLING_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLING_RATE)
        except:
            wav = ffmpeg_load(path)
        wav = pad(wav.float())
        return wav, utt_id