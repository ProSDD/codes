import os
import random
import subprocess
import tempfile
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from RawBoost import (
    ISD_additive_noise,
    LnL_convolutive_noise,
    SSI_additive_noise,
    normWav,
)
SAMPLING_RATE = 16000
TARGET_SAMPLES = 64000  # 4.0s

###RawBoost###
def process_Rawboost_feature(feature, sr, args, algo: int):
    if algo == 1:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
    elif algo == 2:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    elif algo == 3:
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, sr
        )
    elif algo == 4:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, sr
        )
    elif algo == 5:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    elif algo == 6:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, sr
        )
    elif algo == 7:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, sr
        )
    elif algo == 8:
        feature1 = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)
        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)
    return feature


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
    pad_total = max_len - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return F.pad(wav, (pad_left, pad_right))

def load_audio(path: str, target_sr: int = SAMPLING_RATE, max_len: int = TARGET_SAMPLES) -> torch.Tensor:
    try:
        wav, sr = torchaudio.load(path)
    except Exception:
        wav, sr = _load_audio_with_ffmpeg(path, target_sr)

    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    wav = pad(wav, max_len)
    return wav

def load_utt_spk_label(list_path: str) -> Tuple[List[str], List[str], List[int]]:
    utt_ids, spk_ids, labels = [], [], []
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Malformed line: {line}")
            spk = parts[0]
            utt = parts[1]
            lab = parts[4].lower()
            utt_ids.append(utt)
            spk_ids.append(spk)
            labels.append(1 if lab == "bonafide" else 0)

    return utt_ids, spk_ids, labels

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
    vecs = [np.fromstring(fr, sep=",", dtype=np.float32) for fr in frames]
    pros = np.stack(vecs, axis=0)  # [T, 256]
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

class ProSDDStage2Dataset(Dataset):
    def __init__(
        self,
        utt_ids: List[str],
        spk_ids: List[str],
        labels: List[int],
        wav_dir: str,
        spkmean_txt: str,      # spk_id_str -> 192
        prosody_txt: str,      # utt_id -> (T,256)
        sr: int = SAMPLING_RATE,
        max_len: int = TARGET_SAMPLES,
        audio_ext: str = ".flac",
        augment_fn=None,
        augment_algo: int = 0,
        augment_prob: float = 0.0,
        aug_args=None,
    ):
        assert len(utt_ids) == len(spk_ids) == len(labels)
        self.utt_ids = utt_ids
        self.spk_ids = spk_ids
        self.labels = labels
        self.wav_dir = wav_dir
        self.sr = sr
        self.max_len = max_len
        self.audio_ext = audio_ext
        uniq_spk = sorted(set(self.spk_ids))
        self.spk2idx = {s: i for i, s in enumerate(uniq_spk)}
        self.spk2emb = load_spk_mean_embeddings(spkmean_txt)
        self.utt2pros = load_prosody_dict(prosody_txt, 256)
        self.augment_fn = augment_fn
        self.augment_algo = int(augment_algo)
        self.augment_prob = float(augment_prob)
        self.aug_args = aug_args

    def __len__(self):
        return len(self.utt_ids)

    def _maybe_augment(self, wav: torch.Tensor) -> torch.Tensor:
        if self.augment_fn is None or self.augment_algo == 0:
            return wav
        if random.random() > self.augment_prob:
            return wav

        wav_np = wav.detach().cpu().numpy().astype("float32")
        try:
            aug_np = self.augment_fn(wav_np, self.sr, self.aug_args, self.augment_algo).astype("float32")
            aug_t = torch.tensor(aug_np, dtype=torch.float32)
        except Exception as e:
            print(f"[WARN] RawBoost failed: {e}", flush=True)
            return wav

        L = wav.numel()
        if aug_t.numel() < L:
            out = torch.zeros_like(wav)
            out[:aug_t.numel()] = aug_t
            return out
        return aug_t[:L]

    def __getitem__(self, idx: int):
        utt_id = self.utt_ids[idx]
        spk_id_str = self.spk_ids[idx]
        label = int(self.labels[idx])

        spk_idx = int(self.spk2idx[spk_id_str])

        if spk_id_str not in self.spk2emb:
            print(f"[SKIP] Missing spk emb for spkID={spk_id_str} (utt={utt_id})", flush=True)
            return None
        spk_emb = self.spk2emb[spk_id_str]  # do NOT renorm
       

        if utt_id not in self.utt2pros:
            print(f"[SKIP] Missing prosody emb for utt={utt_id}", flush=True)
            return None
        pros_emb = self.utt2pros[utt_id]    # (T,256)

        wav_path = os.path.join(self.wav_dir, utt_id + self.audio_ext)
        try:
            wav = load_audio(wav_path, self.sr, self.max_len)
        except Exception as e:
            print(f"[SKIP] Audio load failed for {wav_path}: {e}", flush=True)
            return None
        wav = self._maybe_augment(wav)

        return wav, spk_emb, pros_emb, spk_idx, label

def collate_stage2(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None 
    
    wavs, spks, pross, spk_idxs, labels = zip(*batch)

    wav = torch.stack(wavs, dim=0)                # (B,L)
    spk = torch.stack(spks, dim=0)                # (B,192)
    spk_idx = torch.tensor(spk_idxs, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    # pad prosody
    lens = [p.size(0) for p in pross]
    Tmax = max(lens)
    D = pross[0].size(1)
    pros = torch.zeros(len(pross), Tmax, D, dtype=pross[0].dtype)
    for i, p in enumerate(pross):
        pros[i, : p.size(0), :] = p
    return wav, spk, pros, spk_idx, labels
