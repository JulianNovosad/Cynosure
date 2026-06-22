import os
import sysconfig
from pathlib import Path
import torch

_TRITON_NVIDIA_DIR = None
_candidate = Path(sysconfig.get_paths().get("platlib", "")) / "triton" / "backends" / "nvidia"
if not (_candidate / "bin" / "ptxas.exe").exists():
    import site
    _candidate = Path(site.getusersitepackages()) / "triton" / "backends" / "nvidia"
if (_candidate / "bin" / "ptxas.exe").exists():
    _TRITON_NVIDIA_DIR = _candidate

if _TRITON_NVIDIA_DIR is not None:
    if "CUDA_PATH" not in os.environ:
        os.environ["CUDA_PATH"] = str(_TRITON_NVIDIA_DIR)

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    pass

if _HAS_TRITON:
    try:
        import triton.windows_utils
        triton.windows_utils.find_cuda.cache_clear()
    except (ImportError, AttributeError):
        pass

    @triton.jit
    def _fused_clamp_nan_kernel(
        x_ptr, n,
        MIN_VAL: tl.constexpr, MAX_VAL: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        x = tl.where(x != x, 0.0, x)
        x = tl.where(x == float('inf'), 0.0, x)
        x = tl.where(x == float('-inf'), 0.0, x)
        x = tl.minimum(tl.maximum(x, MIN_VAL), MAX_VAL)
        tl.store(x_ptr + offsets, x, mask=mask)

    @triton.jit
    def _propagation_kernel(
        activations_ptr, out_targets_ptr, out_weights_ptr,
        frontier_ptr, n_frontier,
        gamma_t, p_start, p_end,
        BLOCK_SIZE: tl.constexpr, MAX_OUT_DEGREE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_frontier

        src_indices = tl.load(frontier_ptr + offsets, mask=mask)
        src_acts = tl.load(activations_ptr + src_indices, mask=mask)

        for k in range(MAX_OUT_DEGREE):
            edge_offsets = src_indices * MAX_OUT_DEGREE + k
            tgts = tl.load(out_targets_ptr + edge_offsets, mask=mask, other=-1)
            valid = (tgts >= 0) & mask

            weights_h = tl.load(out_weights_ptr + edge_offsets, mask=mask, other=0.0)
            weights_f = weights_h.to(tl.float32)
            contribs = src_acts * weights_f

            contribs = tl.where(contribs != contribs, 0.0, contribs)
            contribs = tl.where(contribs == float('inf'), 0.0, contribs)
            contribs = tl.where(contribs == float('-inf'), 0.0, contribs)

            is_pcz = (tgts >= p_start) & (tgts < p_end)
            is_excitatory = weights_f > 0.0
            apply_pcz = is_pcz & is_excitatory & valid
            contribs = tl.where(apply_pcz, contribs * gamma_t, contribs)

            contribs = tl.where(valid, contribs, 0.0)

            tl.atomic_add(activations_ptr + tgts, contribs, mask=valid)

    _KERNEL_CACHE = {}
    _CLAMP_NAN_CACHE = None

    def _get_or_compile_kernel(max_out_degree):
        key = f"prop_{max_out_degree}"
        if key not in _KERNEL_CACHE:
            _KERNEL_CACHE[key] = _propagation_kernel
        return _KERNEL_CACHE[key]

    def _get_clamp_nan_kernel():
        global _CLAMP_NAN_CACHE
        if _CLAMP_NAN_CACHE is None:
            _CLAMP_NAN_CACHE = _fused_clamp_nan_kernel
        return _CLAMP_NAN_CACHE


def triton_available():
    return _HAS_TRITON


def fused_activations_clamp_nan(x, min_val=0.0, max_val=1e4, use_triton=False):
    if _HAS_TRITON and use_triton:
        n = x.numel()
        BLOCK_SIZE = 4096
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        kernel = _get_clamp_nan_kernel()
        kernel[grid](
            x, n,
            MIN_VAL=min_val, MAX_VAL=max_val,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        x.clamp_(min=min_val, max=max_val)
        x.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)


def propagation_step(
    activations, out_targets, out_weights,
    frontier, theta, gamma_t, p_start, p_end,
    use_triton=False,
):
    if frontier.numel() == 0:
        return activations, frontier

    act_before = activations[frontier].clone()

    if _HAS_TRITON and use_triton and activations.is_cuda:
        n_frontier = frontier.shape[0]
        BLOCK_SIZE = 128
        grid = (triton.cdiv(n_frontier, BLOCK_SIZE),)
        kernel = _get_or_compile_kernel(out_targets.shape[1])
        gamma_val = gamma_t.item() if isinstance(gamma_t, torch.Tensor) else gamma_t
        kernel[grid](
            activations, out_targets, out_weights,
            frontier, n_frontier,
            gamma_val, p_start, p_end,
            BLOCK_SIZE=BLOCK_SIZE,
            MAX_OUT_DEGREE=out_targets.shape[1],
        )
    else:
        _python_fallback(activations, out_targets, out_weights, frontier, gamma_t, p_start, p_end)

    fused_activations_clamp_nan(activations, min_val=0.0, max_val=1e4, use_triton=(use_triton and activations.is_cuda))

    crossed_now = activations[frontier] > theta
    crossed_before = act_before > theta
    crossed = crossed_now & ~crossed_before
    new_frontier = frontier[crossed]

    return activations, new_frontier


def _python_fallback(activations, out_targets, out_weights, frontier, gamma_t, p_start, p_end):
    src_targets = out_targets[frontier]
    src_weights = out_weights[frontier]
    src_vals = activations[frontier].unsqueeze(1)

    valid = src_targets >= 0
    if not valid.any():
        return

    excitatory = src_weights > 0
    inhibitory = src_weights < 0

    if excitatory.any():
        all_targets_e = src_targets[valid & excitatory]
        contrib_e = (src_vals * src_weights)[valid & excitatory]

        is_pcz_target = (all_targets_e >= p_start) & (all_targets_e < p_end)
        contrib_e[is_pcz_target] *= gamma_t

        contrib_e.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        activations.scatter_add_(0, all_targets_e, contrib_e)

    if inhibitory.any():
        all_targets_i = src_targets[valid & inhibitory]
        contrib_i = (src_vals * src_weights)[valid & inhibitory]
        contrib_i.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        activations.scatter_add_(0, all_targets_i, contrib_i)
