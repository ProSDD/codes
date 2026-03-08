import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from model_stage1real import ProSDDStage1
from data_utils_stage1real import ProSDDStage1Dataset
from core_scripts.startup_config import set_random_seed

def train_epoch(loader, model, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0

    for wav, spk_emb, prosody_emb, spk_ids in tqdm(loader, desc="Training", leave=False):
        wav = wav.to(device)
        spk_emb = spk_emb.to(device)
        prosody_emb = prosody_emb.to(device)
        spk_ids = torch.as_tensor(spk_ids, device=device)
        loss = model(wav, spk_emb, prosody_emb, spk_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = wav.size(0)
        total_loss += loss.item() * bs
        total += bs
    return total_loss / max(total, 1)

@torch.no_grad()
def validate(loader, model, device):
    model.eval()
    total_loss = 0.0
    total = 0

    for wav, spk_emb, prosody_emb, spk_ids in tqdm(loader, desc="Validating", leave=False):
        wav = wav.to(device)
        spk_emb = spk_emb.to(device)
        prosody_emb = prosody_emb.to(device)
        spk_ids = torch.as_tensor(spk_ids, device=device)
        loss = model(wav, spk_emb, prosody_emb, spk_ids)
        bs = wav.size(0)
        total_loss += loss.item() * bs
        total += bs
    return total_loss / max(total, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # -------- data --------
    parser.add_argument("--train_prosody_txt", type=str, required=True)
    parser.add_argument("--dev_prosody_txt", type=str, required=True)
    parser.add_argument("--train_spkmean_txt", type=str, required=True)
    parser.add_argument("--dev_spkmean_txt", type=str, required=True)
    parser.add_argument("--wav_dir_train", type=str, required=True)
    parser.add_argument("--wav_dir_dev", type=str, required=True)
    parser.add_argument("--audio_ext", type=str, default=".flac")

    # -------- training --------
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ssl_lr", type=float, default=1e-6,
                        help="LR for XLS-R backbone")
    parser.add_argument("--head_lr", type=float, default=1e-4,
                        help="LR for new linear head and other randomly init params")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_workers", type=int, default=0)

    # -------- ProSDD params --------
    parser.add_argument("--mask_prob", type=float, default=0.25)
    parser.add_argument("--mask_span_len", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--log_dir", type=str, default="logs_stage1contrastive")

    args = parser.parse_args()
    set_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # datasets
    train_dataset = ProSDDStage1Dataset(
        prosody_txt=args.train_prosody_txt,
        spkmean_txt=args.train_spkmean_txt,
        wav_dir=args.wav_dir_train,
        audio_ext=args.audio_ext,
    )

    dev_dataset = ProSDDStage1Dataset(
        prosody_txt=args.dev_prosody_txt,
        spkmean_txt=args.dev_spkmean_txt,
        wav_dir=args.wav_dir_dev,
        audio_ext=args.audio_ext,
    )

    print(f"Train samples: {len(train_dataset)}", flush=True)
    print(f"Dev samples:   {len(dev_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # model
    model = ProSDDStage1(
        mask_prob=args.mask_prob,
        mask_span_len=args.mask_span_len,
        tau=args.tau,
    ).to(device)

    # collect parameters for separate LRs
    ssl_param_names = []
    ssl_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # everything under model.ssl.* gets ssl_lr
        if name.startswith("ssl."):
            ssl_params.append(p)
            ssl_param_names.append(name)
        else:
            head_params.append(p)

    print(f"SSL params: {len(ssl_params)}  Head params: {len(head_params)}", flush=True)

    optimizer = torch.optim.AdamW(
        [
            {"params": ssl_params, "lr": args.ssl_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(train_loader, model, optimizer, device)
        val_loss = validate(dev_loader, model, device)

        writer.add_scalar("loss/train_contrastive", train_loss, epoch)
        writer.add_scalar("loss/val_contrastive", val_loss, epoch)

        print(f"Epoch {epoch:03d} | Train={train_loss:.6f} | Val={val_loss:.6f}", flush=True)

        torch.save(
            model.state_dict(),
            os.path.join(args.log_dir, f"model_epoch_{epoch}.pth")
        )