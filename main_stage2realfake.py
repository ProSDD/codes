import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model_stage2realfake import ProSDDStage2
from data_utils_stage2realfake import (
    ProSDDStage2Dataset,
    load_utt_spk_label,
    process_Rawboost_feature,
    collate_stage2,
    SAMPLING_RATE,
    TARGET_SAMPLES,
)
from core_scripts.startup_config import set_random_seed


def train_epoch(loader, model, optimizer, device, epoch, freeze_epochs, alpha, beta, criterion_cls):
    model.train()

    if epoch < freeze_epochs:
        for p in model.cls_head.parameters():
            p.requires_grad = False
        if hasattr(model, "attn"):
            for p in model.attn.parameters():
                p.requires_grad = False
        if hasattr(model, "attn_q"):
            model.attn_q.requires_grad = False
    else:
        for p in model.cls_head.parameters():
            p.requires_grad = True
        if hasattr(model, "attn"):
            for p in model.attn.parameters():
                p.requires_grad = True
        if hasattr(model, "attn_q"):
            model.attn_q.requires_grad = True

    total_loss = total_ssl = total_cls = 0.0
    total_spk_cos = total_pros_cos = 0.0
    total = 0

    for wav, spk_emb, pros_emb, spk_ids, labels in tqdm(loader, desc=f"Training (epoch {epoch})", leave=False):
        wav = wav.to(device)
        spk_emb = spk_emb.to(device)
        pros_emb = pros_emb.to(device)
        spk_ids = spk_ids.to(device)
        labels = labels.to(device)

        out = model(wav, spk_emb, pros_emb, spk_ids)

        ssl_loss = out["ssl_loss"]
        logits = out["logits"]
        cls_loss = criterion_cls(logits, labels)

        if epoch < freeze_epochs:
            loss = beta * ssl_loss
        else:
            loss = alpha * cls_loss + beta * ssl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = wav.size(0)
        total += bs
        total_loss += loss.item() * bs
        total_ssl += float(ssl_loss.item()) * bs
        total_cls += float(cls_loss.item()) * bs
        total_spk_cos += float(out["spk_cos"]) * bs
        total_pros_cos += float(out["pros_cos"]) * bs

    return (
        total_loss / (total + 1e-9),
        total_ssl / (total + 1e-9),
        total_cls / (total + 1e-9),
        total_spk_cos / (total + 1e-9),
        total_pros_cos / (total + 1e-9),
    )


@torch.no_grad()
def validate(loader, model, device, alpha, beta, criterion_cls):
    model.eval()

    total_loss = total_ssl = total_cls = 0.0
    total_spk_cos = total_pros_cos = 0.0
    correct = 0
    total = 0

    tp0 = tp1 = 0
    n0 = n1 = 0

    for wav, spk_emb, pros_emb, spk_ids, labels in tqdm(loader, desc="Validating", leave=False):
        wav = wav.to(device)
        spk_emb = spk_emb.to(device)
        pros_emb = pros_emb.to(device)
        spk_ids = spk_ids.to(device)
        labels = labels.to(device)

        out = model(wav, spk_emb, pros_emb, spk_ids)

        ssl_loss = out["ssl_loss"]
        logits = out["logits"]
        cls_loss = criterion_cls(logits, labels)
        loss = alpha * cls_loss + beta * ssl_loss

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

        mask0 = (labels == 0)
        mask1 = (labels == 1)
        n0 += mask0.sum().item()
        n1 += mask1.sum().item()
        tp0 += ((preds == 0) & mask0).sum().item()
        tp1 += ((preds == 1) & mask1).sum().item()

        bs = wav.size(0)
        total += bs
        total_loss += loss.item() * bs
        total_ssl += float(ssl_loss.item()) * bs
        total_cls += float(cls_loss.item()) * bs
        total_spk_cos += float(out["spk_cos"]) * bs
        total_pros_cos += float(out["pros_cos"]) * bs

    acc = correct / (total + 1e-9)
    acc_spoof = tp0 / (n0 + 1e-9)
    acc_bona = tp1 / (n1 + 1e-9)
    bal_acc = 0.5 * (acc_spoof + acc_bona)

    return (
        total_loss / (total + 1e-9),
        total_ssl / (total + 1e-9),
        total_cls / (total + 1e-9),
        acc,
        acc_bona,
        acc_spoof,
        bal_acc,
        total_spk_cos / (total + 1e-9),
        total_pros_cos / (total + 1e-9),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # -------- data --------
    parser.add_argument("--train_list", type=str, required=True)
    parser.add_argument("--dev_list", type=str, required=True)
    parser.add_argument("--wav_dir_train", type=str, required=True)
    parser.add_argument("--wav_dir_dev", type=str, required=True)
    parser.add_argument("--spkmean_txt_train", type=str, required=True)   # spk -> 192
    parser.add_argument("--prosody_txt_train", type=str, required=True)   # utt -> (T,256)
    parser.add_argument("--spkmean_txt_dev", type=str, required=True)
    parser.add_argument("--prosody_txt_dev", type=str, required=True)

    parser.add_argument("--stage1_ckpt", type=str, default=None)

    # -------- training --------
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--freeze_epochs", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)  # cls weight
    parser.add_argument("--beta", type=float)    # ssl weight
    parser.add_argument("--seed", type=int, default=1234)

    # -------- ProSDD --------
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--mask_span_len", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--num_time_neg", type=int, default=50)
    parser.add_argument("--num_spk_neg", type=int, default=50)
    parser.add_argument("--T_target", type=int, default=200)
    parser.add_argument("--classifier_pool", type=str, default="mean", choices=["mean", "attn"])

    # -------- discriminative LR --------
    parser.add_argument("--lr_ssl_backbone", type=float, default=1e-6)
    parser.add_argument("--lr_ssl_head", type=float, default=1e-4)
    parser.add_argument("--lr_cls", type=float, default=1e-5)

    # -------- RawBoost augmentation (train only) --------
    parser.add_argument("--algo", type=int, default=3)
    parser.add_argument("--augment_prob", type=float, default=0.5)

    parser.add_argument("--nBands", type=int, default=5)
    parser.add_argument("--minF", type=int, default=20)
    parser.add_argument("--maxF", type=int, default=8000)
    parser.add_argument("--minBW", type=int, default=100)
    parser.add_argument("--maxBW", type=int, default=1000)
    parser.add_argument("--minCoeff", type=int, default=10)
    parser.add_argument("--maxCoeff", type=int, default=100)
    parser.add_argument("--minG", type=int, default=0)
    parser.add_argument("--maxG", type=int, default=0)
    parser.add_argument("--minBiasLinNonLin", type=int, default=5)
    parser.add_argument("--maxBiasLinNonLin", type=int, default=20)
    parser.add_argument("--N_f", type=int, default=5)
    parser.add_argument("--P", type=int, default=10)
    parser.add_argument("--g_sd", type=int, default=2)
    parser.add_argument("--SNRmin", type=int, default=10)
    parser.add_argument("--SNRmax", type=int, default=40)

    parser.add_argument("--log_dir", type=str, default="logs_stage2realfake")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--audio_ext", type=str, default=".flac")

    args = parser.parse_args()
    set_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"Audio: sr={SAMPLING_RATE}, samples={TARGET_SAMPLES}", flush=True)

    # load protocol lists
    train_utts, train_spks, train_labels = load_utt_spk_label(args.train_list)
    dev_utts, dev_spks, dev_labels = load_utt_spk_label(args.dev_list)

    # dataset
    train_dataset = ProSDDStage2Dataset(
        utt_ids=train_utts,
        spk_ids=train_spks,
        labels=train_labels,
        wav_dir=args.wav_dir_train,
        spkmean_txt=args.spkmean_txt_train,
        prosody_txt=args.prosody_txt_train,
        sr=SAMPLING_RATE,
        max_len=TARGET_SAMPLES,
        audio_ext=args.audio_ext,
        augment_fn=process_Rawboost_feature if args.algo != 0 else None,
        augment_algo=args.algo,
        augment_prob=args.augment_prob,
        aug_args=args,
    )

    dev_dataset = ProSDDStage2Dataset(
        utt_ids=dev_utts,
        spk_ids=dev_spks,
        labels=dev_labels,
        wav_dir=args.wav_dir_dev,
        spkmean_txt=args.spkmean_txt_dev,
        prosody_txt=args.prosody_txt_dev,
        sr=SAMPLING_RATE,
        max_len=TARGET_SAMPLES,
        audio_ext=args.audio_ext,
        augment_fn=None,
        augment_algo=0,
        augment_prob=0.0,
        aug_args=None,
    )

    print(f"Train samples: {len(train_dataset)}", flush=True)
    print(f"Dev samples:   {len(dev_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_stage2,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_stage2,
    )

    # model
    model = ProSDDStage2(
        mask_prob=args.mask_prob,
        mask_span_len=args.mask_span_len,
        tau=args.tau,
        num_classes=2,
        stage1_ckpt=args.stage1_ckpt,
        num_time_neg=args.num_time_neg,
        num_spk_neg=args.num_spk_neg,
        T_target=args.T_target,
        classifier_pool=args.classifier_pool,
    ).to(device)

    # param groups
    ssl_backbone_params, ssl_head_params, cls_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("ssl."):
            ssl_backbone_params.append(p)
        elif n.startswith("final_proj") or n.startswith("mask_embed") or n.startswith("pros_ln"):
            ssl_head_params.append(p)
        else:
            cls_params.append(p)

    print(
        f"param groups: backbone={len(ssl_backbone_params)} ssl_head={len(ssl_head_params)} cls={len(cls_params)}",
        flush=True
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": ssl_backbone_params, "lr": args.lr_ssl_backbone},
            {"params": ssl_head_params, "lr": args.lr_ssl_head},
            {"params": cls_params, "lr": args.lr_cls},
        ],
        weight_decay=args.weight_decay,
    )

    # class imbalance loss
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion_cls = torch.nn.CrossEntropyLoss(weight=weight)
    print("Using class weights:", weight.tolist(), flush=True)

    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    for epoch in range(1, args.epochs + 1):
        
        if epoch <= 4:
            beta_eff = 0.2
        else:
            beta_eff = 0.05  

        train_loss, train_ssl, train_cls, train_spk_cos, train_pros_cos = train_epoch(
            train_loader,
            model,
            optimizer,
            device,
            epoch,
            args.freeze_epochs,
            args.alpha,
            beta_eff,         
            criterion_cls,
        )

        val_loss, val_ssl, val_cls, val_acc, val_acc_bona, val_acc_spoof, val_bal, val_spk_cos, val_pros_cos = validate(
            dev_loader,
            model,
            device,
            args.alpha,
            beta_eff,          
            criterion_cls,
        )

        writer.add_scalar("loss/train_total", train_loss, epoch)
        writer.add_scalar("loss/train_ssl", train_ssl, epoch)
        writer.add_scalar("loss/train_cls", train_cls, epoch)
        writer.add_scalar("cos/train_spk", train_spk_cos, epoch)
        writer.add_scalar("cos/train_pros", train_pros_cos, epoch)

        writer.add_scalar("loss/val_total", val_loss, epoch)
        writer.add_scalar("loss/val_ssl", val_ssl, epoch)
        writer.add_scalar("loss/val_cls", val_cls, epoch)
        writer.add_scalar("cos/val_spk", val_spk_cos, epoch)
        writer.add_scalar("cos/val_pros", val_pros_cos, epoch)

        writer.add_scalar("acc/val", val_acc, epoch)
        writer.add_scalar("acc/val_bonafide", val_acc_bona, epoch)
        writer.add_scalar("acc/val_spoof", val_acc_spoof, epoch)
        writer.add_scalar("acc/val_balanced", val_bal, epoch)

        print(
            f"Epoch {epoch:03d} | beta={beta_eff:.3f} | "
            f"Train L={train_loss:.4f} (SSL={train_ssl:.4f}, CLS={train_cls:.4f}) | "
            f"Val L={val_loss:.4f} (SSL={val_ssl:.4f}, CLS={val_cls:.4f}) | "
            f"Acc={val_acc:.2%} | Bona={val_acc_bona:.2%} | Spoof={val_acc_spoof:.2%} | Bal={val_bal:.2%} | "
            f"Cos(spk/pros) train={train_spk_cos:.3f}/{train_pros_cos:.3f} val={val_spk_cos:.3f}/{val_pros_cos:.3f}",
            flush=True,
        )
        torch.save(model.state_dict(), os.path.join(args.log_dir, f"model_epoch_{epoch}.pth"))
