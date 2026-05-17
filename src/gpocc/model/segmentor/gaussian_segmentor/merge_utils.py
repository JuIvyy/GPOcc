import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffGaussianUpdaterSparse(nn.Module):
    """
    Learnable differentiable update on sparse bipartite edges (A<->B).
    A: global gaussians (NA)
    B: current frame gaussians (NB)
    edges: idx_A (K,), idx_B (K,) where each edge connects A[idx_A[k]] with B[idx_B[k]]
    """
    def __init__(self, tau=0.3, use_sem=True, alpha_init=0.2):
        super().__init__()
        self.tau = float(tau)
        self.use_sem = bool(use_sem)
        # alpha gate (scalar for now; you can upgrade to per-A gate MLP later)
        self.alpha_logit = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(alpha_init)))))

    def forward(self, A, B, idx_A, idx_B):
        """
        A, B are GaussianPrediction-like objects with fields:
          means: (NA,3)/(NB,3)
          scales: (NA,3)/(NB,3)
          rotations: (NA,4)/(NB,4)  (quaternion)
          opacities: (NA,1)/(NB,1)
          semantics: (NA,C)/(NB,C)
        idx_A, idx_B: (K,)
        return: A_new, b_max_aff (NB,)
        """
        eps = 1e-8
        NA = A.means.shape[0]
        NB = B.means.shape[0]

        a_i = idx_A
        b_j = idx_B

        # --- edge affinity ---
        # distance term
        d2 = (A.means[a_i] - B.means[b_j]).pow(2).sum(dim=-1)          # (K,)
        aff = torch.exp(-d2 / (self.tau + eps))                        # (K,)

        # opacity term (prefer confident B)
        b_opa = B.opacities[b_j].squeeze(-1).clamp(min=0.0)            # (K,)
        aff = aff * b_opa

        # semantic similarity term (optional)
        if self.use_sem:
            a_sem = F.normalize(A.semantics[a_i], dim=-1)
            b_sem = F.normalize(B.semantics[b_j], dim=-1)
            sim = (a_sem * b_sem).sum(dim=-1).clamp(min=0.0)           # (K,)
            aff = aff * sim

        # --- normalize weights per A ---
        # denom[a] = sum_{edges with this a} aff
        denom = torch.zeros((NA,), device=aff.device, dtype=aff.dtype)
        denom.scatter_add_(0, a_i, aff)
        w = aff / (denom[a_i] + eps)                                   # (K,)

        # --- aggregate B to A_hat ---
        mean_hat = torch.zeros_like(A.means)
        scale_hat = torch.zeros_like(A.scales)
        sem_hat = torch.zeros_like(A.semantics)
        opa_hat = torch.zeros_like(A.opacities)
        rot_hat = torch.zeros_like(A.rotations)

        mean_hat.scatter_add_(0, a_i[:, None].expand(-1, 3), w[:, None] * B.means[b_j])
        scale_hat.scatter_add_(0, a_i[:, None].expand(-1, 3), w[:, None] * B.scales[b_j])
        sem_hat.scatter_add_(0, a_i[:, None].expand(-1, A.semantics.shape[-1]), w[:, None] * B.semantics[b_j])
        opa_hat.scatter_add_(0, a_i[:, None].expand(-1, 1), w[:, None] * B.opacities[b_j])

        # quaternion: weighted sum then normalize (simple & works)
        rot_hat.scatter_add_(0, a_i[:, None].expand(-1, 4), w[:, None] * B.rotations[b_j])
        rot_hat = rot_hat / (rot_hat.norm(dim=-1, keepdim=True) + eps)

        # only update A nodes that have any neighbor
        has_nb = denom > 0

        alpha = torch.sigmoid(self.alpha_logit).to(dtype=A.means.dtype)

        means_new = A.means.clone()
        scales_new = A.scales.clone()
        sem_new = A.semantics.clone()
        opa_new = A.opacities.clone()
        rot_new = A.rotations.clone()

        means_new[has_nb] = (1 - alpha) * A.means[has_nb] + alpha * mean_hat[has_nb]
        scales_new[has_nb] = (1 - alpha) * A.scales[has_nb] + alpha * scale_hat[has_nb]
        sem_new[has_nb] = (1 - alpha) * A.semantics[has_nb] + alpha * sem_hat[has_nb]
        opa_new[has_nb] = (1 - alpha) * A.opacities[has_nb] + alpha * opa_hat[has_nb]
        rot_new[has_nb] = (1 - alpha) * A.rotations[has_nb] + alpha * rot_hat[has_nb]
        rot_new = rot_new / (rot_new.norm(dim=-1, keepdim=True) + eps)

        # --- B absorption score (for optional birth) ---
        # max affinity per B among connected edges
        b_max_aff = torch.zeros((NB,), device=aff.device, dtype=aff.dtype)
        b_max_aff.scatter_reduce_(0, b_j, aff, reduce="amax", include_self=True)

        A_new = type(A)(
            means=means_new, scales=scales_new, rotations=rot_new,
            opacities=opa_new, semantics=sem_new,
            feat=getattr(A, "feat", None), conf=getattr(A, "conf", None)
        )
        return A_new, b_max_aff



class DiffGaussianUpdaterSparsePerA(nn.Module):
    """
    Differentiable sparse update with per-A gate alpha_i.

    A: global gaussians (NA)
    B: current-frame gaussians (NB)
    edges: idx_A (K,), idx_B (K,) from radius(pos_B, pos_A, r=...)

    Update:
      aff_e = exp(-||A_i - B_j||^2 / tau) * opa_B * max(0, cos(semA, semB))
      w_e   = aff_e / sum_{e: same A_i} aff_e
      Ahat_i = sum w_e * B_j
      alpha_i = sigmoid( gate_mlp( feat_i ) )
      Anew_i = (1-alpha_i)*A_i + alpha_i*Ahat_i
    """
    def __init__(
        self,
        tau=0.3,
        use_sem=True,
        alpha_init=0.10,      # 初始更新强度（推荐 0.05~0.2）
        gate_hidden=64,
        eps=1e-8,
    ):
        super().__init__()
        self.tau = float(tau)
        self.use_sem = bool(use_sem)
        self.eps = float(eps)

        # gate features (per-A):
        # [opa_A, ent_A, mean_scale_A, log_denom, log_cnt, avg_d2]
        in_dim = 6
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 1),
        )

        # ---- init: 让 gate 初始几乎是常数 alpha_init，且对输入不敏感（更稳）----
        # 这样一开始等价于“固定 EMA 更新”，训练后再学成 per-A
        nn.init.zeros_(self.gate_mlp[-1].weight)
        init_logit = torch.logit(torch.tensor(alpha_init).clamp(1e-4, 1 - 1e-4))
        nn.init.constant_(self.gate_mlp[-1].bias, float(init_logit))

    @staticmethod
    def _sem_entropy(logits: torch.Tensor, eps=1e-8):
        """
        logits: (N,C)
        return entropy in [0,1] roughly (normalized by log C)
        """
        p = F.softmax(logits, dim=-1)
        ent = -(p * (p + eps).log()).sum(dim=-1, keepdim=True)
        C = logits.shape[-1]
        ent = ent / (torch.log(torch.tensor(float(C), device=logits.device)) + eps)
        return ent.clamp(0.0, 1.0)

    def forward(self, A, B, idx_A, idx_B):
        """
        A, B: GaussianPrediction-like objects (no leading batch dim here)
          A.means: (NA,3), A.scales:(NA,3), A.rotations:(NA,4), A.opacities:(NA,1), A.semantics:(NA,C)
          B.* similarly with NB

        idx_A, idx_B: (K,) Long tensors
        return:
          A_new (same type as A),
          b_max_aff (NB,)  # for optional birth heuristic
        """
        eps = self.eps
        NA = A.means.shape[0]
        NB = B.means.shape[0]

        a_i = idx_A
        b_j = idx_B

        # ---------- edge affinity ----------
        # distance
        d2 = (A.means[a_i] - B.means[b_j]).pow(2).sum(dim=-1)                    # (K,)
        aff = torch.exp(-d2 / (self.tau + eps))                                   # (K,)

        # opacity of B
        b_opa = B.opacities[b_j].squeeze(-1).clamp(min=0.0)                       # (K,)
        aff = aff * b_opa

        # semantic similarity (optional)
        if self.use_sem:
            a_sem = F.normalize(A.semantics[a_i], dim=-1)
            b_sem = F.normalize(B.semantics[b_j], dim=-1)
            sim = (a_sem * b_sem).sum(dim=-1).clamp(min=0.0)                      # (K,)
            aff = aff * sim

        # ---------- per-A denom / counts / avg stats ----------
        denom = torch.zeros((NA,), device=aff.device, dtype=aff.dtype)
        denom.scatter_add_(0, a_i, aff)                                            # sum aff per A

        cnt = torch.zeros((NA,), device=aff.device, dtype=aff.dtype)
        cnt.scatter_add_(0, a_i, torch.ones_like(aff))                             # neighbor count per A

        # avg d2 per A (weighted by aff, then / denom)
        d2_sum = torch.zeros((NA,), device=aff.device, dtype=aff.dtype)
        d2_sum.scatter_add_(0, a_i, aff * d2)
        avg_d2 = d2_sum / (denom + eps)                                            # (NA,)

        has_nb = denom > 0

        # ---------- normalize edge weights ----------
        w = aff / (denom[a_i] + eps)                                               # (K,)

        # ---------- aggregate B -> A_hat ----------
        mean_hat = torch.zeros_like(A.means)
        scale_hat = torch.zeros_like(A.scales)
        sem_hat = torch.zeros_like(A.semantics)
        opa_hat = torch.zeros_like(A.opacities)
        rot_hat = torch.zeros_like(A.rotations)

        mean_hat.scatter_add_(0, a_i[:, None].expand(-1, 3), w[:, None] * B.means[b_j])
        scale_hat.scatter_add_(0, a_i[:, None].expand(-1, 3), w[:, None] * B.scales[b_j])
        sem_hat.scatter_add_(0, a_i[:, None].expand(-1, A.semantics.shape[-1]), w[:, None] * B.semantics[b_j])
        opa_hat.scatter_add_(0, a_i[:, None].expand(-1, 1), w[:, None] * B.opacities[b_j])

        rot_hat.scatter_add_(0, a_i[:, None].expand(-1, 4), w[:, None] * B.rotations[b_j])
        rot_hat = rot_hat / (rot_hat.norm(dim=-1, keepdim=True) + eps)

        # ---------- per-A gate alpha_i ----------
        opa_A = A.opacities.squeeze(-1).clamp(0.0, 1.0)                            # (NA,)
        ent_A = self._sem_entropy(A.semantics, eps=eps).squeeze(-1)                # (NA,)
        mean_scale_A = A.scales.mean(dim=-1)                                       # (NA,)
        log_denom = torch.log(denom + eps)                                         # (NA,)
        log_cnt = torch.log(cnt + 1.0)                                             # (NA,)

        gate_feat = torch.stack(
            [opa_A, ent_A, mean_scale_A, log_denom, log_cnt, avg_d2],
            dim=-1
        )                                                                          # (NA,6)

        alpha_i = torch.sigmoid(self.gate_mlp(gate_feat)).squeeze(-1)              # (NA,)
        alpha_i = alpha_i * has_nb.to(alpha_i.dtype)                               # no neighbor => alpha=0

        # ---------- update A ----------
        means_new = A.means + alpha_i[:, None] * (mean_hat - A.means)
        scales_new = A.scales + alpha_i[:, None] * (scale_hat - A.scales)
        sem_new = A.semantics + alpha_i[:, None] * (sem_hat - A.semantics)
        opa_new = A.opacities + alpha_i[:, None] * (opa_hat - A.opacities)

        rot_new = A.rotations + alpha_i[:, None] * (rot_hat - A.rotations)
        rot_new = rot_new / (rot_new.norm(dim=-1, keepdim=True) + eps)

        A_new = type(A)(
            means=means_new,
            scales=scales_new,
            rotations=rot_new,
            opacities=opa_new,
            semantics=sem_new,
            feat=getattr(A, "feat", None),
            conf=getattr(A, "conf", None),
        )

        # ---------- B absorption score (optional birth heuristic) ----------
        b_max_aff = torch.zeros((NB,), device=aff.device, dtype=aff.dtype)
        b_max_aff.scatter_reduce_(0, b_j, aff, reduce="amax", include_self=True)

        return A_new, b_max_aff

