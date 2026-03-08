import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from typing import Optional


class ProSDDStage2(nn.Module):

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-xls-r-300m",
        mask_prob: float = 0.25,
        mask_span_len: int = 8,
        tau: float = 0.07,
        num_classes: int = 2,
        stage1_ckpt: Optional[str] = None,
        num_time_neg: int = 50,
        num_spk_neg: int = 50,
        T_target: int = 200,
        classifier_pool: str = "mean",  
    ):
        super().__init__()

        self.mask_prob = float(mask_prob)
        self.mask_span_len = int(mask_span_len)
        self.tau = float(tau)

        self.spk_dim = 192
        self.prosody_dim = 256
        self.out_dim = self.spk_dim + self.prosody_dim  # 448
        self.T_target = int(T_target)

        self.num_time_neg = int(num_time_neg)
        self.num_spk_neg = int(num_spk_neg)

        self.ssl = Wav2Vec2Model.from_pretrained(
            model_name,
            output_hidden_states=False,
            output_attentions=False,
        )
        self.hidden_dim = self.ssl.config.hidden_size  # 1024

        self.mask_embed = nn.Parameter(torch.zeros(self.hidden_dim))
        nn.init.normal_(self.mask_embed, mean=0.0, std=0.02)

        self.pros_ln = nn.LayerNorm(self.prosody_dim)

        # 1024 -> 448 (Stage-1 head)
        self.final_proj = nn.Linear(self.hidden_dim, self.out_dim)

        # classifier head (on clean ctx)
        self.classifier_pool = classifier_pool
        if classifier_pool == "attn":
            self.attn = nn.MultiheadAttention(self.hidden_dim, num_heads=8, batch_first=True)
            self.attn_q = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            nn.init.normal_(self.attn_q, mean=0.0, std=0.02)

        self.cls_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

        if stage1_ckpt is not None:
            self.load_stage1(stage1_ckpt)

    def load_stage1(self, ckpt_path: str):
        print(f"Loading Stage-1 weights from {ckpt_path}", flush=True)
        state = torch.load(ckpt_path, map_location="cpu")

        
        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k.replace("module.", "")
            new_state[k] = v

        # ssl
        ssl_keys = {k.replace("ssl.", ""): v for k, v in new_state.items() if k.startswith("ssl.")}
        self.ssl.load_state_dict(ssl_keys, strict=False)

        # mask
        if "mask_embed" in new_state:
            self.mask_embed.data.copy_(new_state["mask_embed"])

        # projection head
        if "final_proj.weight" in new_state and "final_proj.bias" in new_state:
            self.final_proj.load_state_dict(
                {"weight": new_state["final_proj.weight"], "bias": new_state["final_proj.bias"]},
                strict=True,
            )

        print("Stage-1 weights loaded into Stage-2 (ssl/mask/final_proj).", flush=True)

    def _compute_span_mask(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        num_to_mask = int(self.mask_prob * T)
        num_to_mask = max(1, min(num_to_mask, T))

        for b in range(B):
            masked = 0
            while masked < num_to_mask:
                start = torch.randint(0, T, (1,), device=device).item()
                end = min(start + self.mask_span_len, T)
                newly_masked = (~mask[b, start:end]).sum().item()
                mask[b, start:end] = True
                masked += newly_masked

            if not mask[b].any():
                t = torch.randint(0, T, (1,), device=device)
                mask[b, t] = True

        return mask

    def _contrastive_loss_with_metrics(self, pred, target, mask, spk_ids):
        idx = mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            z = pred.new_tensor(0.0)
            return z, z, z

        b, t = idx[:, 0], idx[:, 1]
        N = b.size(0)
        B, T, D = target.shape
        device = target.device

        p = F.normalize(pred[b, t], dim=-1)
        pos = F.normalize(target[b, t], dim=-1)

        with torch.no_grad():
            p_spk, p_pros = p[:, :self.spk_dim], p[:, self.spk_dim:]
            pos_spk, pos_pros = pos[:, :self.spk_dim], pos[:, self.spk_dim:]
            spk_cos = F.cosine_similarity(p_spk, pos_spk, dim=-1).mean()
            pros_cos = F.cosine_similarity(p_pros, pos_pros, dim=-1).mean()
            print(f"pos cosine | speaker: {spk_cos.item():.3f}, prosody: {pros_cos.item():.3f}",flush=True)

        # Neg A: same utt, different time
        Kt = self.num_time_neg
        t_neg = torch.randint(0, T, (N, Kt), device=device)
        t_true = t.unsqueeze(1).expand_as(t_neg)
        same_t = (t_neg == t_true)
        if same_t.any():
            t_neg[same_t] = (t_neg[same_t] + 1) % T
        neg_time = F.normalize(target[b.unsqueeze(1), t_neg], dim=-1)  # (N,Kt,D)

        # Neg B: different speaker, same time
        spk_ids_tensor = torch.as_tensor(spk_ids, device=device)
        max_spk_neg = max(0, B - 1)
        Ks = min(self.num_spk_neg, max_spk_neg)

        if Ks == 0:
            neg_spk = target.new_empty((N, 0, D))
        else:
            all_idx = torch.arange(B, device=device)
            b_neg = torch.empty((N, Ks), dtype=torch.long, device=device)
            valid_counts = torch.zeros(N, dtype=torch.long, device=device)

            for i in range(N):
                anchor_b = b[i].item()
                anchor_spk = spk_ids_tensor[anchor_b]
                candidates = all_idx[all_idx != anchor_b]
                candidates = candidates[spk_ids_tensor[candidates] != anchor_spk]

                if candidates.numel() == 0:
                    valid_counts[i] = 0
                    b_neg[i].fill_(anchor_b)
                    continue

                k = min(Ks, candidates.numel())
                perm = torch.randperm(candidates.numel(), device=device)
                chosen = candidates[perm[:k]]

                if k < Ks:
                    pad = chosen[torch.randint(0, k, (Ks - k,), device=device)]
                    chosen = torch.cat([chosen, pad], dim=0)

                b_neg[i] = chosen
                valid_counts[i] = Ks

            neg_spk = F.normalize(target[b_neg, t.unsqueeze(1)], dim=-1)  # (N,Ks,D)
            no_cands = (valid_counts == 0)
            if no_cands.any():
                neg_spk[no_cands] = 0.0

        pos_sim = torch.sum(p * pos, dim=-1, keepdim=True)
        time_sim = torch.einsum("nd,nkd->nk", p, neg_time)
        spk_sim = torch.einsum("nd,nkd->nk", p, neg_spk)

        logits = torch.cat([pos_sim, time_sim, spk_sim], dim=1) / self.tau
        labels = torch.zeros(N, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)

        return loss, spk_cos.detach(), pros_cos.detach()

    def forward(self, wav, spk_emb, prosody_emb, spk_ids):
        # feature extractor
        z = self.ssl.feature_extractor(wav)
        if isinstance(z, dict):
            z = z["input_values"]
        elif isinstance(z, (tuple, list)):
            z = z[0]
        z = z.transpose(1, 2)  # (B,C,T')

        # feature projection
        z = self.ssl.feature_projection(z)
        if isinstance(z, (tuple, list)):
            z = z[0]  # (B,T,H)

        B, T, H = z.shape
        device = z.device

        # enforce fixed T
        Tt = self.T_target
        if T > Tt:
            z = z[:, :Tt, :]
        elif T < Tt:
            z = torch.cat([z, z.new_zeros(B, Tt - T, H)], dim=1)
        T = z.size(1)

        # align prosody to T
        Tp = prosody_emb.size(1)
        if Tp < T:
            pad = prosody_emb.new_zeros(B, T - Tp, prosody_emb.size(-1))
            prosody_emb = torch.cat([prosody_emb, pad], dim=1)
        elif Tp > T:
            prosody_emb = prosody_emb[:, :T, :]

        # build GT target (B,T,448)
        spk = spk_emb.unsqueeze(1).expand(B, T, self.spk_dim)
        prosody_n = self.pros_ln(prosody_emb)
        target = torch.cat([spk, prosody_n], dim=-1)

        # PASS 1 masked
        mask = self._compute_span_mask(B, T, device=device)
        z_masked = z.clone()
        z_masked[mask] = self.mask_embed.to(device)

        out_masked = self.ssl.encoder(
            z_masked,
            attention_mask=None,
            output_hidden_states=False,
            return_dict=True,
        )
        ctx_masked = out_masked.last_hidden_state
        pred = self.final_proj(ctx_masked)
        ssl_loss, spk_cos, pros_cos = self._contrastive_loss_with_metrics(pred, target, mask, spk_ids)

        # PASS 2 clean classifier
        out_clean = self.ssl.encoder(
            z,
            attention_mask=None,
            output_hidden_states=False,
            return_dict=True,
        )
        ctx_clean = out_clean.last_hidden_state

        if self.classifier_pool == "attn":
            q = self.attn_q.expand(B, -1, -1)
            attn_out, _ = self.attn(q, ctx_clean, ctx_clean, need_weights=False)
            pooled = attn_out.squeeze(1)
        else:
            pooled = ctx_clean.mean(dim=1)

        logits = self.cls_head(pooled)

        return {
            "logits": logits,
            "ssl_loss": ssl_loss,
            "spk_cos": spk_cos,
            "pros_cos": pros_cos,
        }