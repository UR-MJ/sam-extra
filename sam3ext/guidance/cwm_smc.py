"""CFG Wavelet Mixing and Sliding-Mode Control guidance transforms."""

from __future__ import annotations

from typing import Any

import torch

from .haar import (
    haar_dwt2d,
    haar_idwt2d,
    pad_even,
    safe_compute_dtype,
    sigma_norm,
)


SMC_NORM_EPS = 1e-8
SMC_DELTA_FLOOR = 1e-8

# Keep these names and values aligned with namemechan/ComfyUI-DCW.  ``Auto``
# resolves to one of the concrete presets at generation start, after Forge has
# attached the real model to the processing object.
SMC_PRESETS: dict[str, tuple[float, float]] = {
    "SD1.5 / SD2": (5.0, 0.10),
    "SDXL": (5.0, 0.10),
    "SD3 / SD3.5": (6.0, 0.10),
    "Flux": (6.0, 0.70),
    "Qwen-Image": (6.0, 0.10),
    "Cosmos / Wan": (6.0, 0.20),
    "Custom": (6.0, 0.10),
}
SMC_PRESET_NAMES: tuple[str, ...] = (
    "Off",
    "Auto",
    *SMC_PRESETS.keys(),
)


def normalize_smc_preset(value: Any) -> str:
    """Return a public SMC preset name, defaulting unknown input to ``Off``."""
    text = str(value or "").strip().casefold()
    aliases = {name.casefold(): name for name in SMC_PRESET_NAMES}
    return aliases.get(text, "Off")


def _model_candidates(model: Any) -> list[Any]:
    """Collect the small Forge/Comfy model wrapper chain without recursion."""
    if model is None:
        return []
    result: list[Any] = []
    pending = [model]
    seen: set[int] = set()
    while pending and len(result) < 24:
        current = pending.pop(0)
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(current)
        for name in (
            "forge_objects",
            "unet",
            "model",
            "diffusion_model",
            "model_config",
        ):
            child = getattr(current, name, None)
            if child is not None and id(child) not in seen:
                pending.append(child)
    return result


def detect_smc_preset(model: Any) -> str:
    """Match ComfyUI-DCW's ``Auto`` model-family detection in Forge wrappers."""
    candidates = _model_candidates(model)
    names = " ".join(type(item).__name__.casefold() for item in candidates)

    # Preserve upstream's class-name priority.  Forge's Anima engine exposes
    # the top-level class as ``Anima`` and its DiT as a Cosmos/Predict2 family.
    if "flux" in names:
        return "Flux"
    if any(token in names for token in ("cosmos", "predict2", "wan", "anima")):
        return "Cosmos / Wan"
    if any(token in names for token in ("sd3", "mmdit")):
        return "SD3 / SD3.5"
    if "sdxl" in names:
        return "SDXL"
    if "qwen" in names:
        return "Qwen-Image"

    model_types = " ".join(
        str(getattr(item, "model_type", "") or "").casefold()
        for item in candidates
    )
    if "flow" in model_types:
        return "SD3 / SD3.5"
    if any(token in model_types for token in ("v_pred", "v-pred", "vprediction")):
        return "SD1.5 / SD2"
    return "SD1.5 / SD2"


def resolve_smc_preset(
    preset: Any,
    model: Any = None,
    *,
    custom_lambda: float = 6.0,
    custom_k: float = 0.10,
) -> tuple[str, str, float, float]:
    """Resolve ``(requested, concrete, lambda, k)`` for an SMC selection."""
    requested = normalize_smc_preset(preset)
    if requested == "Off":
        return requested, "Off", 0.0, 0.0
    concrete = detect_smc_preset(model) if requested == "Auto" else requested
    if concrete == "Custom":
        return requested, concrete, float(custom_lambda), float(custom_k)
    lambda_value, k_value = SMC_PRESETS[concrete]
    return requested, concrete, lambda_value, k_value


def _finite_or_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def apply_smc_error(
    error: torch.Tensor,
    previous,
    lambda_value: float,
    k_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply unit-L2 switching control and return ``(corrected, new_state)``."""
    # Match ComfyUI-DCW's public contract: either zero disables SMC. Although
    # lambda=0 could define a derivative-only controller mathematically, the
    # reference node deliberately uses it as an off sentinel.
    if lambda_value == 0.0 or k_value == 0.0:
        return error, error.detach()

    working = _finite_or_zero(error.float())
    if previous is None or previous.shape != working.shape:
        previous_working = working.detach()
    else:
        previous_working = _finite_or_zero(
            previous.to(device=working.device, dtype=working.dtype)
        )
    surface = _finite_or_zero(
        (working - previous_working) + float(lambda_value) * previous_working
    )
    reduce_dims = tuple(range(1, surface.ndim))
    norm = torch.linalg.vector_norm(
        surface, dim=reduce_dims, keepdim=True
    ).clamp_min(SMC_NORM_EPS)
    raw_delta = -float(k_value) * (surface / norm)
    delta_limit = (
        0.5 * working.abs().mean(dim=reduce_dims, keepdim=True)
    ).clamp_min(SMC_DELTA_FLOOR)
    delta = raw_delta.clamp(-delta_limit, delta_limit)
    corrected = _finite_or_zero(working + delta)
    return corrected.to(error.dtype), corrected.detach()


def apply_cwm_error(
    error: torch.Tensor,
    sigma,
    effective_scale: float,
    alpha_low: float,
    alpha_high: float,
) -> torch.Tensor:
    """Scale CFG error by scheduled Haar bands."""
    scale = float(effective_scale)
    if alpha_low == 0.0 and alpha_high == 0.0:
        return error * scale

    original_dtype = error.dtype
    working = _finite_or_zero(
        error.to(dtype=safe_compute_dtype(original_dtype))
    )
    schedule = sigma_norm(sigma, working)
    low_scale = scale * (1.0 + float(alpha_low) * schedule)
    high_scale = scale * (1.0 + float(alpha_high) * (1.0 - schedule))
    # Negative products have no real geometric mean. Fall back to an
    # arithmetic midpoint rather than emitting NaNs for experimental values.
    product = low_scale * high_scale
    middle_scale = torch.where(
        product >= 0,
        product.clamp_min(0).sqrt(),
        (low_scale + high_scale) * 0.5,
    ) if torch.is_tensor(product) else (
        product**0.5 if product >= 0 else (low_scale + high_scale) * 0.5
    )

    padded, (height, width) = pad_even(working)
    ll, lh, hl, hh = haar_dwt2d(padded)
    result = haar_idwt2d(
        ll * low_scale,
        lh * middle_scale,
        hl * middle_scale,
        hh * high_scale,
    )[..., :height, :width]
    return result.to(original_dtype)


def compose_cfg(
    cond: torch.Tensor,
    uncond: torch.Tensor,
    sigma,
    effective_scale: float,
    mode: str,
    alpha_low: float,
    alpha_high: float,
    smc_lambda: float,
    smc_k: float,
    smc_previous,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compose standard/CWM/SMC/SMC+CWM in denoised space."""
    error = _finite_or_zero(cond.float() - uncond.float())
    uncond_working = _finite_or_zero(uncond.float())
    next_previous = smc_previous
    if mode in {"smc", "smc+cwm"}:
        error, next_previous = apply_smc_error(
            error, smc_previous, smc_lambda, smc_k
        )
    if mode in {"cwm", "smc+cwm"}:
        guided_error = apply_cwm_error(
            error, sigma, effective_scale, alpha_low, alpha_high
        )
    else:
        guided_error = error * float(effective_scale)
    return _finite_or_zero(uncond_working + guided_error), next_previous
