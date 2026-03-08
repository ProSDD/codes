import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

class ProSDDStage1(nn.Module):
    def __init__(
        self,
        model_name="facebook/wav2vec2-xls-r-300m",
        mask_prob=0.25,
        mask_span_len=8,
        tau=0.07,
        out_dim=448,
        num_time_neg=50,
        num_spk_neg=50,
    ):
        super().__init__()

        self.mask_prob = mask_prob
        self.mask_span_len = mask_span_len
        self.tau = tau

        self.spk_dim = 192
        self.prosody_dim = 256
        self.out_dim = self.spk_dim + self.prosody_dim

        self.pros_ln = nn.LayerNorm(self.prosody_dim)

        self.num_time_neg = int(num_time_neg)
        self.num_spk_neg = int(num_spk_neg)

        # backbone
        self.ssl = Wav2Vec2Model.from_pretrained(
            model_name,
            output_hidden_states=False,
            output_attentions=False,
        )
        self.hidden_dim = self.ssl.config.hidden_size  # 1024

        # learned mask embedding
        self.mask_embed = nn.Parameter(torch.zeros(self.hidden_dim))
        nn.init.normal_(self.mask_embed, mean=0.0, std=0.02)

        # project 1024 -> 448
        self.final_proj = nn.Linear(self.hidden_dim, out_dim)


    ### Masking ###
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
                tt = torch.randint(0, T, (1,), device=device)
                mask[b, tt] = True

        return mask

    ### Contrastive loss ###
    # 50 time + 50 speaker
    def _contrastive_loss(self, pred, target, mask, spk_ids):
        """
        pred:   (B, T, 448)
        target: (B, T, 448) = [spk | prosody]
        mask:   (B, T) bool
        spk_ids: (B,)
        """
        idx = mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return pred.new_tensor(0.0)

        b, t = idx[:, 0], idx[:, 1]  # (N,)
        N = b.size(0)
        B, T, D = target.shape
        device = target.device

        # ====================================================
        # Positives
        # ====================================================
        p = F.normalize(pred[b, t], dim=-1)       # (N, D)
        pos = F.normalize(target[b, t], dim=-1)   # (N, D)

        with torch.no_grad():
            p_spk, p_pros = p[:, :self.spk_dim], p[:, self.spk_dim:]
            pos_spk, pos_pros = pos[:, :self.spk_dim], pos[:, self.spk_dim:]
            spk_cos = F.cosine_similarity(p_spk, pos_spk, dim=-1).mean()
            pros_cos = F.cosine_similarity(p_pros, pos_pros, dim=-1).mean()
            print(f"pos cosine | speaker: {spk_cos.item():.3f}, prosody: {pros_cos.item():.3f}",flush=True)

        # ====================================================
        # Negatives A: same utterance (same speaker), different time
        # ====================================================
        num_time_neg = int(self.num_time_neg)
        t_neg = torch.randint(0, T, (N, num_time_neg), device=device)

        # avoid sampling the true time index
        t_true = t.unsqueeze(1).expand_as(t_neg)
        same_t = (t_neg == t_true)
        if same_t.any():
            t_neg[same_t] = (t_neg[same_t] + 1) % T

        neg_time = F.normalize(target[b.unsqueeze(1), t_neg], dim=-1)  # (N, num_time_neg, D)

        # ====================================================
        # Negatives B: different speaker, same time
        # ====================================================
        spk_ids_tensor = torch.as_tensor(spk_ids, device=device)  # (B,)

        max_spk_neg = max(0, B - 1)
        num_spk_neg = min(int(self.num_spk_neg), max_spk_neg)

        if num_spk_neg == 0:
            neg_spk = target.new_empty((N, 0, D))
        else:
            all_idx = torch.arange(B, device=device)

            b_neg = torch.empty((N, num_spk_neg), dtype=torch.long, device=device)
            valid_counts = torch.zeros(N, dtype=torch.long, device=device)

            for i in range(N):
                anchor_b = b[i].item()
                anchor_spk = spk_ids_tensor[anchor_b]

                candidates = all_idx[all_idx != anchor_b]
                candidates = candidates[spk_ids_tensor[candidates] != anchor_spk]

                if candidates.numel() == 0:
                    valid_counts[i] = 0
                    b_neg[i].fill_(anchor_b)  # safe init
                    continue

                k = min(num_spk_neg, candidates.numel())
                perm = torch.randperm(candidates.numel(), device=device)
                chosen = candidates[perm[:k]]

                if k < num_spk_neg:
                    pad = chosen[torch.randint(0, k, (num_spk_neg - k,), device=device)]
                    chosen = torch.cat([chosen, pad], dim=0)

                b_neg[i] = chosen
                valid_counts[i] = num_spk_neg

            neg_spk = F.normalize(target[b_neg, t.unsqueeze(1)], dim=-1)

            # Debug 2: verify speaker negatives are truly different-speaker
            with torch.no_grad():
                frac_diff = (
                    spk_ids_tensor[b_neg] != spk_ids_tensor[b].unsqueeze(1)
                ).float().mean()
                print("fraction truly diff-speaker in speaker negatives:", frac_diff.item(),flush=True)

            no_cands = (valid_counts == 0)
            if no_cands.any():
                neg_spk[no_cands] = 0.0

        # ====================================================
        # logits: [pos | time negs | spk negs]
        # ====================================================
        pos_sim = torch.sum(p * pos, dim=-1, keepdim=True)          # (N, 1)
        time_sim = torch.einsum("nd,nkd->nk", p, neg_time)          # (N, num_time_neg)
        spk_sim = torch.einsum("nd,nkd->nk", p, neg_spk)            # (N, num_spk_neg)

        logits = torch.cat([pos_sim, time_sim, spk_sim], dim=1)
        logits = logits / self.tau

        labels = torch.zeros(N, dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)

   
    ### Forward ###
    def forward(self, wav, spk_emb, prosody_emb, spk_ids):
        """
        wav:         (B, samples)
        spk_emb:     (B, 192)
        prosody_emb: (B, T', 256)
        spk_ids:     (B,)
        """
        #feature encoder
        z = self.ssl.feature_extractor(wav)
        if isinstance(z, dict):
            z = z["input_values"]
        elif isinstance(z, (tuple, list)):
            z = z[0]
        z = z.transpose(1, 2)  # (B, C, T')

        #feature projection
        z = self.ssl.feature_projection(z)
        if isinstance(z, (tuple, list)):
            z = z[0]  # (B, T, 1024)

        B, T, H = z.shape
        device = z.device

        #enforce exactly 200 frames
        T_target = 200
        if T > T_target:
            z = z[:, :T_target, :]
        elif T < T_target:
            z = torch.cat([z, z.new_zeros(B, T_target - T, H)], dim=1)

        T = z.size(1)

        #prosody length to T
        Tp = prosody_emb.size(1)
        if Tp < T:
            pad = prosody_emb.new_zeros(B, T - Tp, prosody_emb.size(-1))
            prosody_emb = torch.cat([prosody_emb, pad], dim=1)
        elif Tp > T:
            prosody_emb = prosody_emb[:, :T, :]

        #build GT targets (B,T,448) = [spk(192) | prosody(256)]
        spk = spk_emb.unsqueeze(1).expand(B, T, spk_emb.size(-1))
        prosody_emb = self.pros_ln(prosody_emb)
        target = torch.cat([spk, prosody_emb], dim=-1)

        #mask latent features
        mask = self._compute_span_mask(B, T, device=device)
        z_masked = z.clone()
        z_masked[mask] = self.mask_embed.to(device)

        #transformer
        outputs = self.ssl.encoder(
            z_masked,
            attention_mask=None,
            output_hidden_states=False,
            return_dict=True
        )
        ctx = outputs.last_hidden_state  # (B,T,1024)

        #project 1024 -> 448
        pred = self.final_proj(ctx)      # (B,T,448)

        #contrastive loss
        return self._contrastive_loss(pred, target, mask, spk_ids)
