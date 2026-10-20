# trasmuon/optimizer.py
"""
TrasMuon Optimizer (v2.1)
-------------------------
Standalone TrasMuon optimizer pipeline:
- Supports DDP and DTensor row-sharded (FSDP2 row shard only; no last-dim shard)
- Newton–Schulz: Triton kernel when (CUDA && ns_iters==5 && use_triton), else compiled ref
- All optimizer state fp32; update cast back to param dtype
- eta_hat computed from pre-gate ||O_base||_F so gate is absolute damping
- Low host-sync design: step counters are tensors, gate updates use update_mask

v2.1 Fixes (vs v2):
[P1] Default trigger mode (use_energy_trigger=True) + safer defaults:
     - Only gate columns with r_j > energy_k; others stay at 1.0
     - energy_k default -> 3.0
     - Fixes soft-gate floor_rate≈1 issue when cmin too high
     - Adds r_star diagnostic: r_star = exp((1/cmin - 1)/alpha) - 1
[P2] _compute_energy_gate_logform signature fixed:
     - Takes col_energy as input (no redundant compute)
     - Returns exactly 3 values: (c_j, active_rate, floor_rate)
     - floor_rate in trigger mode is computed among active columns only
[P3] Eliminated CPU sync in eta_hat gamma path:
     - All gamma computations are pure tensor ops (no .item() in step hot path)
[P4] Added sf_exclude_warmup (default True):
     - When True: S_lr2/C_lr2 do NOT accumulate during warmup
     - S0 auto-calibration occurs at warmup_steps + sf_warmup_steps

Algorithm (2D W in R^{out x in}):
1) g32 = grad.float()
2) m32 <- beta1*m32 + (1-beta1)*g32
3) O = NS(m32)                      # orthogonalize
4) v_row <- beta2*v_row + (1-beta2)*mean_cols(O^2)
5) O_base = O32 * (1/(sqrt(v_row)+eps))[:,None]
6) eta_hat = scale_factor * lr * sqrt(out*in) / ||O_base||_F
7) Energy-only gate (log-form) on X_source in {m,g,O_base}:
   col_energy = sum_i X_ij^2 (global across row shards)
   E_ref_ema updated EVERY step (EMA of quantile(col_energy))
   r_j = col_energy / (E_ref_ema + eps_energy)
   c_raw = 1/(1 + alpha*log(r_j+1))
   c_j = clamp(c_raw, cmin, 1) (optionally trigger-mask)
   c_ema <- gate_ema*c_ema + (1-gate_ema)*c_j  (only every col_gate_every steps, via mask)
8) Schedule-free averaging:
   gamma = lr or clamp(eta_hat around eta_hat_ema)
   if sf_exclude_warmup:
       accumulate only when step > warmup_steps
       S0 auto-calibration at warmup_steps + sf_warmup_steps
   else:
       accumulate all steps (warmup uses c_last_eff=1)
   c_avg = C_lr2 / S_lr2
   c_used = (1-sf_mix)*c_ema + sf_mix*c_avg
   warmup_steps: if in warmup -> c_used=1
9) W <- W - lr*wd*W - eta_hat*(O_base * c_used)[cast]

Non-2D params: AdamW fallback (fp32 second moment, tensor bias-correction)

Date: 2026-01-18 (v2.1 standalone cleanup)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT

# Reuse the repo's NS Triton kernel.
from .newton_schulz_triton import newton_schulz_triton


# ============================================================
# Newton–Schulz (variable steps)
# ============================================================
_NS_CONSTS = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]


@torch.compile(dynamic=False, fullgraph=True)
def _zeropower_via_newtonschulz_var(G: Tensor, steps: int, eps: float) -> Tensor:
    """
    Variable-steps Newton–Schulz (fp32-safe path).
    Uses the same constants as repo's 5-step implementation, truncated to `steps`.
    Returns bf16 like repo kernel path.
    """
    steps = int(steps)
    if steps <= 0:
        return G

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm <= 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    for a, b, c in _NS_CONSTS[: min(steps, len(_NS_CONSTS))]:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _ns_orthogonalize_fp32(G: Tensor, *, ns_iters: int, eps: float, use_triton: bool) -> Tensor:
    """
    Orthogonalize G using Newton–Schulz. Returns bf16 tensor.
    - If use_triton and ns_iters==5 on CUDA: use repo Triton kernel.
    - Else: compiled variable-step reference.
    """
    ns_iters = int(ns_iters)
    G = G.contiguous()
    if use_triton and ns_iters == 5 and G.is_cuda:
        return newton_schulz_triton(G, epsilon=float(eps))
    return _zeropower_via_newtonschulz_var(G, steps=ns_iters, eps=float(eps))


# ============================================================
# Helpers
# ============================================================
@torch.no_grad()
def _safe_quantile_kth(x: Tensor, q: float) -> Tensor:
    """
    Quantile via kthvalue (deterministic, avoids interpolation cost of torch.quantile).
    - x must be 1D.
    - q in (0,1].
    """
    q = float(q)
    n = x.numel()
    k = int(q * n + 0.5)
    k = max(1, min(n, k))

    if x.device.type == "mps":
        # MPS may not support kthvalue reliably; fallback to CPU
        return x.detach().cpu().kthvalue(k).values.to(x.device)
    return x.kthvalue(k).values


@torch.no_grad()
def _compute_energy_gate_logform(
    col_energy: Tensor,  # (in_dim,) fp32, already globalized if row-sharded
    E_ref_ema: Tensor,  # scalar fp32 tensor (updated outside)
    *,
    in_dim: int,
    col_gate_alpha: float,
    col_gate_cmin: float,
    eps_energy: float,
    energy_k: float,
    use_energy_trigger: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    [P2] Compute energy-only gate (log-form), returns exactly 3 values:
      c_j: (in_dim,) in [cmin, 1]
      active_rate: scalar fp32 tensor
      floor_rate: scalar fp32 tensor

    Notes:
    - Trigger mode: gate only where r_j > energy_k, others stay 1.0
      floor_rate is computed among active columns only (0 if no active columns).
    - Soft mode: gate all columns; active_rate=1
      floor_rate is computed over all columns.
    """
    device = col_energy.device
    dtype = col_energy.dtype
    in_dim_f = torch.tensor(float(in_dim), device=device, dtype=dtype)

    # ratio r_j = E_j / (E_ref_ema + eps)
    r_j = col_energy / (E_ref_ema + float(eps_energy))

    # log-form damping: c_raw = 1 / (1 + alpha * log(r_j + 1))
    log_term = torch.log(r_j + 1.0)
    c_raw = 1.0 / (1.0 + float(col_gate_alpha) * log_term)
    c_clipped = torch.clamp(c_raw, min=float(col_gate_cmin), max=1.0)

    tiny = torch.tensor(1e-12, device=device, dtype=dtype)

    if use_energy_trigger:
        apply_mask = r_j > float(energy_k)
        c_j = torch.where(apply_mask, c_clipped, torch.ones_like(c_clipped))

        n_active = apply_mask.to(dtype=dtype).sum()
        active_rate = n_active / (in_dim_f + tiny)

        # floor rate among active only
        at_floor_active = ((c_j <= (float(col_gate_cmin) + 1e-6)) & apply_mask).to(dtype=dtype).sum()
        floor_rate = at_floor_active / (n_active + tiny)
    else:
        c_j = c_clipped
        active_rate = torch.ones((), device=device, dtype=dtype)
        floor_rate = (c_j <= (float(col_gate_cmin) + 1e-6)).to(dtype=dtype).mean()

    return c_j, active_rate, floor_rate


# ============================================================
# Optimizer
# ============================================================
class TrasMuon(Optimizer):
    """
    TrasMuon (v2.1) for the trimmed TrasMuon training path.
    Engineering notes: masked updates, minimal host-sync, DTensor row-shard support.
    """

    def __init__(
        self,
        params: ParamsT,
        *,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.95, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        ns_iters: int = 5,
        scale_factor: float = 0.2,
        use_triton: bool = True,
        # Energy-only gate
        col_gate_enabled: bool = True,
        col_gate_use: Literal["m", "g", "O_base"] = "m",
        energy_quantile: float = 0.5,
        eps_energy: float = 1e-12,
        col_gate_alpha: float = 1.0,
        col_gate_cmin: float = 0.7,
        energy_k: float = 3.0,              # [P1] default changed
        use_energy_trigger: bool = True,    # [P1] default changed
        gate_ema: float = 0.9,
        col_gate_every: int = 4,
        col_gate_min_in: int = 8,
        # E_ref EMA
        E_ref_ema_beta: float = 0.99,
        # warmup
        warmup_steps: int = 0,
        # Schedule-free
        sf_enabled: bool = True,
        sf_mix: float = 0.3,
        sf_exclude_warmup: bool = True,     # [P4] new
        gamma_source: Literal["lr", "eta_hat"] = "lr",
        gamma_eta_ema: float = 0.9,
        gamma_eta_lo_mul: float = 0.3,
        gamma_eta_hi_mul: float = 3.0,
        sf_tau_mode: Literal["fixed", "auto"] = "auto",
        sf_t_half: int = 2000,
        sf_warmup_steps: int = 200,
        S0: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if scale_factor <= 0.0:
            raise ValueError(f"Invalid scale_factor: {scale_factor}")
        if col_gate_use not in ("m", "g", "O_base"):
            raise ValueError(f"col_gate_use must be 'm','g','O_base', got: {col_gate_use}")
        if not (0.0 < col_gate_cmin <= 1.0):
            raise ValueError(f"col_gate_cmin must be in (0,1], got: {col_gate_cmin}")
        if not (0.0 <= gate_ema < 1.0):
            raise ValueError(f"gate_ema must be in [0,1), got: {gate_ema}")
        if col_gate_every < 1:
            raise ValueError(f"col_gate_every must be >=1, got: {col_gate_every}")
        if not (0.0 < energy_quantile <= 1.0):
            raise ValueError(f"energy_quantile must be in (0,1], got: {energy_quantile}")
        if gamma_source not in ("lr", "eta_hat"):
            raise ValueError(f"gamma_source must be 'lr' or 'eta_hat', got: {gamma_source}")

        defaults = dict(
            lr=float(lr),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps),
            weight_decay=float(weight_decay),
            ns_iters=int(ns_iters),
            scale_factor=float(scale_factor),
            use_triton=bool(use_triton),
            col_gate_enabled=bool(col_gate_enabled),
            col_gate_use=str(col_gate_use),
            energy_quantile=float(energy_quantile),
            eps_energy=float(eps_energy),
            col_gate_alpha=float(col_gate_alpha),
            col_gate_cmin=float(col_gate_cmin),
            energy_k=float(energy_k),
            use_energy_trigger=bool(use_energy_trigger),
            gate_ema=float(gate_ema),
            col_gate_every=int(col_gate_every),
            col_gate_min_in=int(col_gate_min_in),
            E_ref_ema_beta=float(E_ref_ema_beta),
            warmup_steps=int(warmup_steps),
            sf_enabled=bool(sf_enabled),
            sf_mix=float(sf_mix),
            sf_exclude_warmup=bool(sf_exclude_warmup),
            gamma_source=str(gamma_source),
            gamma_eta_ema=float(gamma_eta_ema),
            gamma_eta_lo_mul=float(gamma_eta_lo_mul),
            gamma_eta_hi_mul=float(gamma_eta_hi_mul),
            sf_tau_mode=str(sf_tau_mode),
            sf_t_half=int(sf_t_half),
            sf_warmup_steps=int(sf_warmup_steps),
            S0=float(S0),
        )
        super().__init__(params, defaults)

        # Distributed wiring for the standalone TrasMuon optimizer path.
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh is supported, got {distributed_mesh.ndim}D. "
                    f"For HSDP, provide the 1D sharded sub-mesh."
                )
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError(
                f"Invalid distributed_mesh type: {type(distributed_mesh)}. Expected DeviceMesh or ProcessGroup."
            )
        self._distributed_mesh = distributed_mesh

        # Optional: per-layer gate enable logging
        self._gate_layer_log: Dict[int, Dict[str, Any]] = {}

    def get_gate_layer_log(self) -> Dict[int, Dict[str, Any]]:
        return self._gate_layer_log.copy()

    def _infer_row_shard_dim(self, p: DTensor) -> Optional[int]:
        """
        Detect DTensor row-sharding along dim -2 only for 2D matrices.
        Disallow last-dim sharding.
        """
        if not isinstance(p, DTensor):
            return None

        shard_placements = [
            (i, pl)
            for i, pl in enumerate(p.placements)
            if pl.is_shard() and p.device_mesh.size(i) > 1
        ]
        if not shard_placements:
            return None
        if len(shard_placements) > 1:
            raise NotImplementedError("TrasMuon does not support multiple sharded dimensions.")

        shard_dim = shard_placements[0][1].dim
        if shard_dim == p.ndim - 1:
            raise NotImplementedError("TrasMuon does not support sharding along last dim (-1).")
        if p.ndim == 2 and shard_dim != (p.ndim - 2):
            raise NotImplementedError("TrasMuon only supports 2D matrices sharded along row dim (-2).")
        return shard_dim

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        pg = self._process_group
        rank = self._device_rank
        world = self._world_size

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]
            ns_iters: int = group["ns_iters"]
            scale_factor: float = group["scale_factor"]
            use_triton: bool = group["use_triton"]

            col_gate_enabled: bool = group["col_gate_enabled"]
            col_gate_use: str = group["col_gate_use"]
            energy_quantile: float = group["energy_quantile"]
            eps_energy: float = group["eps_energy"]
            col_gate_alpha: float = group["col_gate_alpha"]
            col_gate_cmin: float = group["col_gate_cmin"]
            energy_k: float = group["energy_k"]
            use_energy_trigger: bool = group["use_energy_trigger"]
            gate_ema: float = group["gate_ema"]
            col_gate_every: int = group["col_gate_every"]
            col_gate_min_in: int = group["col_gate_min_in"]
            E_ref_ema_beta: float = group["E_ref_ema_beta"]
            warmup_steps: int = group["warmup_steps"]

            sf_enabled: bool = group["sf_enabled"]
            sf_mix: float = group["sf_mix"]
            sf_exclude_warmup: bool = group["sf_exclude_warmup"]
            gamma_source: str = group["gamma_source"]
            gamma_eta_ema: float = group["gamma_eta_ema"]
            gamma_eta_lo_mul: float = group["gamma_eta_lo_mul"]
            gamma_eta_hi_mul: float = group["gamma_eta_hi_mul"]
            sf_tau_mode: str = group["sf_tau_mode"]
            sf_t_half: int = group["sf_t_half"]
            sf_warmup_steps: int = group["sf_warmup_steps"]
            S0_init: float = group["S0"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Local views (DTensor-safe)
                p_local = p.to_local() if isinstance(p, DTensor) else p
                g = p.grad
                g_local = g.to_local() if isinstance(g, DTensor) else g

                param_dtype = p_local.dtype
                device = p_local.device

                state = self.state[p]
                if len(state) == 0:
                    # step counter as tensor (avoid host sync)
                    state["step"] = torch.zeros((), device=device, dtype=torch.int64)

                    # momentum fp32 local shape
                    state["momentum"] = torch.zeros_like(p_local, dtype=torch.float32, device=device)

                    if p_local.ndim == 2:
                        out_dim_local, in_dim_local = p_local.shape

                        # row second moment
                        state["v_row"] = torch.zeros(out_dim_local, device=device, dtype=torch.float32)

                        # gate / SF state
                        state["c_ema"] = torch.ones(in_dim_local, device=device, dtype=torch.float32)
                        state["c_last"] = torch.ones(in_dim_local, device=device, dtype=torch.float32)

                        # E_ref EMA (scalar)
                        state["E_ref_ema"] = torch.zeros((), device=device, dtype=torch.float32)
                        state["E_ref_initialized"] = torch.zeros((), device=device, dtype=torch.int64)

                        # schedule-free
                        if sf_enabled:
                            state["S_lr2"] = torch.zeros((), device=device, dtype=torch.float32)
                            state["C_lr2"] = torch.zeros(in_dim_local, device=device, dtype=torch.float32)
                            state["c_avg"] = torch.ones(in_dim_local, device=device, dtype=torch.float32)

                            state["S0"] = torch.tensor(S0_init, device=device, dtype=torch.float32)
                            state["S0_calibrated"] = torch.zeros((), device=device, dtype=torch.int64)
                            state["gamma2_ema"] = torch.zeros((), device=device, dtype=torch.float32)

                            if gamma_source == "eta_hat":
                                state["eta_hat_ema"] = torch.zeros((), device=device, dtype=torch.float32)

                        # diagnostics (scalar tensors, no host sync)
                        state["eta_hat"] = torch.zeros((), device=device, dtype=torch.float32)
                        state["O_base_fro"] = torch.zeros((), device=device, dtype=torch.float32)
                        state["O_hat_fro"] = torch.zeros((), device=device, dtype=torch.float32)

                        state["c_ema_min"] = torch.ones((), device=device, dtype=torch.float32)
                        state["c_ema_mean"] = torch.ones((), device=device, dtype=torch.float32)

                        state["active_rate"] = torch.zeros((), device=device, dtype=torch.float32)
                        state["floor_rate"] = torch.zeros((), device=device, dtype=torch.float32)
                        state["E_ref_ema_t"] = torch.zeros((), device=device, dtype=torch.float32)

                        state["c_avg_min"] = torch.ones((), device=device, dtype=torch.float32)
                        state["c_avg_mean"] = torch.ones((), device=device, dtype=torch.float32)
                        state["tau"] = torch.zeros((), device=device, dtype=torch.float32)
                        state["gamma_t"] = torch.tensor(lr, device=device, dtype=torch.float32)
                        state["S_lr2_t"] = torch.zeros((), device=device, dtype=torch.float32)
                        state["S0_t"] = torch.tensor(S0_init, device=device, dtype=torch.float32)

                        # [P1] r_star diagnostic (where c_raw == cmin in soft-gate log-form)
                        if col_gate_cmin < 1.0 and col_gate_alpha > 0.0:
                            r_star = math.exp((1.0 / float(col_gate_cmin) - 1.0) / float(col_gate_alpha)) - 1.0
                        else:
                            r_star = float("inf")
                        state["r_star"] = torch.tensor(r_star, device=device, dtype=torch.float32)

                        # per-layer gate enable log (pure python decision)
                        gate_enabled_for_layer = bool(col_gate_enabled) and (int(in_dim_local) >= int(col_gate_min_in))
                        pid = id(p)
                        if gate_enabled_for_layer:
                            reason = "enabled"
                        elif not col_gate_enabled:
                            reason = "col_gate_enabled=False"
                        else:
                            reason = f"in_dim={int(in_dim_local)} < col_gate_min_in={int(col_gate_min_in)}"
                        self._gate_layer_log[pid] = {
                            "enabled": gate_enabled_for_layer,
                            "reason": reason,
                            "shape": (int(p_local.shape[0]), int(p_local.shape[1])),
                        }
                    else:
                        # AdamW fallback exp_avg_sq fp32
                        state["exp_avg_sq"] = torch.zeros_like(p_local, dtype=torch.float32, device=device)

                # step += 1
                state["step"].add_(1)
                step_i: Tensor = state["step"]  # int64 scalar tensor

                # grad fp32
                g32 = g_local.detach().to(dtype=torch.float32)

                # decoupled wd
                if weight_decay != 0.0:
                    p_local.mul_(1.0 - lr * weight_decay)

                # momentum
                m32: Tensor = state["momentum"]
                m32.mul_(beta1).add_(g32, alpha=(1.0 - beta1))

                # 2D: NorMuon + energy gate + SF
                if p_local.ndim == 2:
                    out_dim, in_dim = p.shape  # global dims for DTensor; local dims for regular tensor
                    out_local, in_local = p_local.shape
                    assert in_local == in_dim, "Row-sharded DTensor must keep full column dim locally."

                    # DTensor row-shard detect
                    row_shard_dim = self._infer_row_shard_dim(p) if isinstance(p, DTensor) else None
                    row_sharded = (row_shard_dim is not None) and (pg is not None) and (world > 1)

                    # ----- warmup mask (tensor, no .item()) -----
                    if warmup_steps > 0:
                        warmup_steps_t = torch.tensor(int(warmup_steps), device=device, dtype=torch.int64)
                        in_warmup = (step_i <= warmup_steps_t)  # bool scalar tensor
                    else:
                        in_warmup = torch.zeros((), device=device, dtype=torch.bool)
                    not_warmup = ~in_warmup

                    # ----- NS orthogonalization -----
                    if row_sharded:
                        m_contig = m32.contiguous()
                        gathered = [torch.empty_like(m_contig) for _ in range(world)]
                        dist.all_gather(gathered, m_contig, group=pg)
                        m_full = torch.cat(gathered, dim=row_shard_dim).contiguous()
                        O_full = _ns_orthogonalize_fp32(m_full, ns_iters=ns_iters, eps=eps, use_triton=use_triton)
                        shards = torch.tensor_split(O_full, world, dim=row_shard_dim)
                        O = shards[rank].contiguous()
                    else:
                        O = _ns_orthogonalize_fp32(m32, ns_iters=ns_iters, eps=eps, use_triton=use_triton)

                    O32 = O.to(dtype=torch.float32)

                    # ----- row scaling -----
                    v_row: Tensor = state["v_row"]
                    O_sq = O32 * O32
                    mean_cols_O_sq = O_sq.mean(dim=1)
                    v_row.mul_(beta2).add_(mean_cols_O_sq, alpha=(1.0 - beta2))
                    row_scale = 1.0 / (torch.sqrt(v_row) + eps)
                    O_base = O32 * row_scale.view(out_local, 1)

                    # ----- eta_hat from ||O_base||_F (global if row-sharded) -----
                    fro2_local = (O_base * O_base).sum()
                    if row_sharded:
                        fro2 = fro2_local.clone()
                        dist.all_reduce(fro2, op=dist.ReduceOp.SUM, group=pg)
                    else:
                        fro2 = fro2_local

                    tiny = torch.tensor(1e-12, device=device, dtype=torch.float32)
                    O_base_fro = torch.sqrt(fro2 + torch.tensor(0.0, device=device, dtype=torch.float32))
                    eta_hat = (
                        torch.tensor(scale_factor * lr * math.sqrt(float(out_dim * in_dim)), device=device, dtype=torch.float32)
                        / (O_base_fro + tiny)
                    )
                    state["eta_hat"].copy_(eta_hat)
                    state["O_base_fro"].copy_(O_base_fro)

                    # gate enable (pure python, no sync)
                    gate_enabled = bool(col_gate_enabled) and (int(in_dim) >= int(col_gate_min_in))

                    # gate / SF state
                    c_ema: Tensor = state["c_ema"]
                    c_last: Tensor = state["c_last"]
                    E_ref_ema: Tensor = state["E_ref_ema"]
                    E_ref_initialized: Tensor = state["E_ref_initialized"]

                    # ----- choose X_source for energy -----
                    if col_gate_use == "m":
                        X_source = m32  # fp32 local
                    elif col_gate_use == "g":
                        X_source = g32  # fp32 local
                    else:
                        X_source = O_base  # fp32 local

                    # col_energy local then globalize if row-sharded
                    col_energy = (X_source * X_source).sum(dim=0)  # (in_dim,)
                    if row_sharded:
                        dist.all_reduce(col_energy, op=dist.ReduceOp.SUM, group=pg)

                    # ----- E_ref_ema update EVERY step (masked init + EMA; no .item()) -----
                    E_current = _safe_quantile_kth(col_energy, energy_quantile)
                    is_uninit = (E_ref_initialized == 0)

                    beta = float(E_ref_ema_beta)
                    E_ref_ema_new = E_ref_ema.mul(beta).add(E_current, alpha=(1.0 - beta))
                    E_ref_ema.copy_(torch.where(is_uninit, E_current, E_ref_ema_new))

                    E_ref_initialized.add_(is_uninit.to(dtype=torch.int64))
                    E_ref_initialized.clamp_(0, 1)
                    state["E_ref_ema_t"].copy_(E_ref_ema)

                    # ===== Schedule-free accumulation (v2.1: optional warmup exclusion) =====
                    if sf_enabled:
                        S_lr2: Tensor = state["S_lr2"]
                        C_lr2: Tensor = state["C_lr2"]
                        c_avg: Tensor = state["c_avg"]
                        S0: Tensor = state["S0"]
                        S0_cal: Tensor = state["S0_calibrated"]
                        gamma2_ema: Tensor = state["gamma2_ema"]

                        # gamma tensor (store in state["gamma_t"])
                        if gamma_source == "lr":
                            state["gamma_t"].fill_(float(lr))
                            gamma = state["gamma_t"]
                        else:
                            eta_hat_ema: Tensor = state["eta_hat_ema"]
                            one_i64 = torch.tensor(1, device=device, dtype=torch.int64)
                            is_first = (step_i == one_i64)

                            ema_new = eta_hat_ema.mul(float(gamma_eta_ema)).add(eta_hat, alpha=(1.0 - float(gamma_eta_ema)))
                            eta_hat_ema.copy_(torch.where(is_first, eta_hat, ema_new))

                            lo = eta_hat_ema * float(gamma_eta_lo_mul)
                            hi = eta_hat_ema * float(gamma_eta_hi_mul)
                            gamma = torch.clamp(eta_hat, min=lo, max=hi)
                            state["gamma_t"].copy_(gamma)

                        gamma_sq = gamma * gamma

                        # [P4] accumulate mask: exclude warmup by default
                        if sf_exclude_warmup:
                            acc_mask = not_warmup.to(dtype=torch.float32)  # 1 post-warmup else 0
                        else:
                            acc_mask = torch.ones((), device=device, dtype=torch.float32)

                        # gamma2_ema update (masked; init at first (post-)step)
                        if sf_exclude_warmup and warmup_steps > 0:
                            warmup_steps_t = torch.tensor(int(warmup_steps), device=device, dtype=torch.int64)
                            steps_after = step_i - warmup_steps_t
                            is_first_after = (steps_after == torch.tensor(1, device=device, dtype=torch.int64)) & not_warmup
                            g2_new = gamma2_ema.mul(0.99).add(gamma_sq, alpha=0.01)
                            gamma2_ema_candidate = torch.where(is_first_after, gamma_sq, g2_new)
                            gamma2_ema.copy_(torch.where(not_warmup, gamma2_ema_candidate, gamma2_ema))
                        else:
                            one_i64b = torch.tensor(1, device=device, dtype=torch.int64)
                            is_first2 = (step_i == one_i64b)
                            g2_new = gamma2_ema.mul(0.99).add(gamma_sq, alpha=0.01)
                            gamma2_ema.copy_(torch.where(is_first2, gamma_sq, g2_new))

                        # [P4] S0 auto-calibration point
                        if sf_tau_mode == "auto":
                            if sf_exclude_warmup:
                                cal_step = int(warmup_steps) + int(sf_warmup_steps)
                            else:
                                cal_step = int(sf_warmup_steps)
                            cal_step_t = torch.tensor(cal_step, device=device, dtype=torch.int64)

                            should_cal = (step_i == cal_step_t) & (S0_cal == 0)
                            cal_mask = should_cal.to(dtype=torch.float32)

                            new_S0 = torch.maximum(gamma2_ema * float(sf_t_half), tiny)
                            S0.copy_(S0 + (new_S0 - S0) * cal_mask)

                            S0_cal.add_(should_cal.to(dtype=torch.int64))
                            S0_cal.clamp_(0, 1)
                            state["S0_t"].copy_(S0)

                        # c_last_eff: if including warmup, warmup uses 1.0; if excluding warmup, irrelevant
                        ones_in = torch.ones_like(c_last)
                        if gate_enabled:
                            if sf_exclude_warmup:
                                c_last_eff = c_last
                            else:
                                c_last_eff = torch.where(in_warmup, ones_in, c_last)
                        else:
                            c_last_eff = ones_in

                        # masked accumulation
                        S_lr2.add_(gamma_sq * acc_mask)
                        C_lr2.add_(c_last_eff * gamma_sq * acc_mask)

                        # update c_avg only after S_lr2 becomes non-zero (avoid warmup-exclude 0/0 -> 0)
                        c_avg_new = C_lr2 / (S_lr2 + tiny)
                        c_avg.copy_(torch.where(S_lr2 > 0.0, c_avg_new, c_avg))

                        tau = S_lr2 / (S_lr2 + S0 + tiny)

                        state["c_avg_min"].copy_(c_avg.min())
                        state["c_avg_mean"].copy_(c_avg.mean())
                        state["tau"].copy_(tau)
                        state["S_lr2_t"].copy_(S_lr2)

                    # ===== Gate update (compute every step, masked EMA apply) =====
                    if gate_enabled:
                        k = torch.tensor(int(col_gate_every), device=device, dtype=torch.int64)
                        one = torch.tensor(1, device=device, dtype=torch.int64)
                        should_update = (step_i.remainder(k) == 0) | (step_i == one)

                        update_mask = (should_update & not_warmup).to(dtype=torch.float32)  # scalar fp32

                        c_j, active_rate, floor_rate = _compute_energy_gate_logform(
                            col_energy=col_energy,
                            E_ref_ema=E_ref_ema,
                            in_dim=int(in_dim),
                            col_gate_alpha=float(col_gate_alpha),
                            col_gate_cmin=float(col_gate_cmin),
                            eps_energy=float(eps_energy),
                            energy_k=float(energy_k),
                            use_energy_trigger=bool(use_energy_trigger),
                        )

                        ema_target = c_ema.mul(float(gate_ema)).add(c_j, alpha=(1.0 - float(gate_ema)))
                        c_ema.add_((ema_target - c_ema) * update_mask)

                        state["active_rate"].copy_(active_rate)
                        state["floor_rate"].copy_(floor_rate)
                        state["c_ema_min"].copy_(c_ema.min())
                        state["c_ema_mean"].copy_(c_ema.mean())
                    else:
                        state["active_rate"].fill_(0.0)
                        state["floor_rate"].fill_(0.0)

                    # ===== choose c_used (warmup forces ones) =====
                    ones_in2 = torch.ones_like(c_ema)
                    if gate_enabled:
                        if sf_enabled:
                            c_used_raw = (1.0 - float(sf_mix)) * c_ema + float(sf_mix) * state["c_avg"]
                        else:
                            c_used_raw = c_ema
                        c_used = torch.where(in_warmup, ones_in2, c_used_raw)
                    else:
                        c_used = ones_in2

                    # cache for next step SF
                    c_last.copy_(c_used)

                    # ----- apply gate + update -----
                    O_hat = O_base * c_used.view(1, int(in_dim))

                    fro2h_local = (O_hat * O_hat).sum()
                    if row_sharded:
                        fro2h = fro2h_local.clone()
                        dist.all_reduce(fro2h, op=dist.ReduceOp.SUM, group=pg)
                    else:
                        fro2h = fro2h_local
                    O_hat_fro = torch.sqrt(fro2h + torch.tensor(0.0, device=device, dtype=torch.float32))
                    state["O_hat_fro"].copy_(O_hat_fro)

                    update = (eta_hat * O_hat).to(dtype=param_dtype)
                    p_local.add_(update, alpha=-1.0)

                else:
                    # -----------------------------
                    # Non-2D: AdamW fallback (fp32)
                    # -----------------------------
                    exp_avg_sq: Tensor = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).addcmul_(g32, g32, value=(1.0 - beta2))

                    step_f = state["step"].to(dtype=torch.float32)
                    b1 = torch.tensor(beta1, device=device, dtype=torch.float32)
                    b2 = torch.tensor(beta2, device=device, dtype=torch.float32)

                    bias_correction1 = 1.0 - torch.pow(b1, step_f)
                    bias_correction2 = 1.0 - torch.pow(b2, step_f)

                    m_hat = m32 / bias_correction1
                    v_hat = exp_avg_sq / bias_correction2

                    upd = m_hat / (torch.sqrt(v_hat) + eps)
                    p_local.add_(upd.to(dtype=param_dtype), alpha=-lr)

        return loss
