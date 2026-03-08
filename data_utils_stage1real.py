import os
import subprocess
import tempfile
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


TARGET_SAMPLES = 64000  # 4.0s
def _load_audio_with_ffmpeg(path: str, target_sr: int) -> Tuple[torch.Tensor, int]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        cmd = [
            "ffmpeg", "-y",
            "-i", path,
            "-ac", "1",
            "-ar", str(target_sr),
            "-f", "wav",
            tmp.name,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        wav, sr = torchaudio.load(tmp.name)
    return wav, sr

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
    else:
        pad_total = max_len - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return F.pad(wav, (pad_left, pad_right))
    
def load_audio(path: str, target_sr: int, max_len: int) -> torch.Tensor:
    try:
        wav, sr = torchaudio.load(path)
    except Exception:
        # ffmpeg fallback
        wav, sr = _load_audio_with_ffmpeg(path, target_sr)
        
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)  # (L,)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    wav = pad(wav, max_len)
    return wav

def load_spk_mean_embeddings(spk_txt: str) -> Dict[str, torch.Tensor]:
    spk2emb = {}
    with open(spk_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            spk = parts[0]
            vec = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
            spk2emb[spk] = vec
    return spk2emb

def _parse_prosody_line(line: str) -> Tuple[str, np.ndarray]:
    line = line.strip()
    utt, frames_str = line.split("\t")
    frames = frames_str.split("|")
    # Each frame: "a,b,c,..."
    vecs = [np.fromstring(fr, sep=",", dtype=np.float32) for fr in frames]
    pros = np.stack(vecs, axis=0)  # [T, D]
    return utt, pros

def load_prosody_dict(prosody_txt: str, expected_dim: int = 256) -> Dict[str, torch.Tensor]:
    utt2pros = {}
    with open(prosody_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt, pros = _parse_prosody_line(line)

            if pros.ndim != 2 or pros.shape[1] != expected_dim:
                raise ValueError(
                    f"Prosody dim mismatch for {utt}: got {pros.shape}, expected (*,{expected_dim})"
                )

            utt2pros[utt] = torch.from_numpy(pros)  # float32
    return utt2pros

class ProSDDStage1Dataset(Dataset):
    def __init__(
        self,
        prosody_txt: str,
        spkmean_txt: str,
        wav_dir: str,
        sr: int = 16000,
        max_len: int = 64000,
        audio_ext: str = ".flac",
        prosody_dim: int = 256,
    ):
        super().__init__()

        self.wav_dir = wav_dir
        self.sr = sr
        self.max_len = max_len
        self.audio_ext = audio_ext
        self.prosody_dim = prosody_dim
        self.spk2emb = load_spk_mean_embeddings(spkmean_txt)          # spkID -> (192)
        self.utt2pros = load_prosody_dict(prosody_txt, prosody_dim)   # uttID -> (T,256)
        self.utt_ids = list(self.utt2pros.keys())

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx: int):
        utt_id = self.utt_ids[idx]  # e.g., "103-1240-0000"
        spk_str = utt_id.split("-")[0]  # "103"

        if spk_str not in self.spk2emb:
            raise KeyError(f"Missing speaker mean embedding for spkID={spk_str} (utt={utt_id})")
        spk_id = int(spk_str)

        # embeddings
        spk_emb = self.spk2emb[spk_str]          # No Norm as (192,) already normalized in file
        prosody = self.utt2pros[utt_id]          # (T,256) raw
        wav_path = os.path.join(self.wav_dir, utt_id + self.audio_ext)
        wav = load_audio(wav_path, self.sr, self.max_len)  # (max_len,)

        return wav, spk_emb, prosody, spk_id