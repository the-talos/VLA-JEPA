"""Device helpers shared by the inference path (XPU, CUDA and CPU).

Detection order is XPU, then CUDA, then CPU, and an explicit device argument
always wins. ``torch.cuda.*`` raises on a wheel built without CUDA, so code that
used to call it goes through this module instead.
"""

import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)

DeviceLike = Union[str, int, torch.device, None]
_ACCELERATORS = ("xpu", "cuda")


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def default_device_type() -> str:
    """Best available backend: ``xpu``, then ``cuda``, then ``cpu``."""
    for device_type in _ACCELERATORS:
        if is_available(device_type):
            return device_type
    return "cpu"


def resolve_device(device: DeviceLike = None, index: Optional[int] = None) -> torch.device:
    """Resolve ``None``, a string or a ``torch.device`` to a concrete device.

    ``None`` detects the backend, and ``index`` picks the card on it (ignored on
    CPU). An explicit device is returned as given, so an unsupported one still
    fails loudly instead of being replaced by another backend.
    """
    if isinstance(device, torch.device):
        return device
    if device is not None:
        return torch.device(device)
    device_type = default_device_type()
    if device_type == "cpu" or index is None:
        return torch.device(device_type)
    return torch.device(device_type, int(index))


def device_type(device: Union[str, torch.device, torch.Tensor]) -> str:
    """Backend type (``xpu``, ``cuda`` or ``cpu``) of a device, string or tensor."""
    if isinstance(device, torch.Tensor):
        return device.device.type
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0]


def synchronize(device: Union[str, torch.device, None] = None) -> None:
    """Synchronize the accelerator of ``device`` (default: detected); no-op on CPU."""
    kind = device_type(device) if device is not None else default_device_type()
    if kind in _ACCELERATORS and is_available(kind):
        getattr(torch, kind).synchronize()


def autocast(device: Union[str, torch.device, torch.Tensor, None] = None, dtype: torch.dtype = torch.bfloat16):
    """``torch.autocast`` for the backend of ``device``.

    Autocast stays disabled on CPU, as the previous hard-coded
    ``torch.autocast("cuda", ...)`` was there. XPU autocast only accepts bfloat16
    and float16, so a float32 request is disabled as well; see
    :func:`casts_to_float32`.
    """
    kind = device_type(device) if device is not None else default_device_type()
    enabled = kind == "cuda" or (kind == "xpu" and dtype != torch.float32)
    return torch.autocast(kind, dtype=dtype, enabled=enabled)


def casts_to_float32(device: Union[str, torch.device, torch.Tensor, None] = None) -> bool:
    """Whether ``autocast(device, torch.float32)`` casts inputs to float32.

    Only CUDA does. Elsewhere the caller has to cast the inputs of a float32
    region itself.
    """
    kind = device_type(device) if device is not None else default_device_type()
    return kind == "cuda"


def attn_implementation(device: Union[str, torch.device, None] = None) -> str:
    """``flash_attention_2`` on CUDA (the original default), ``sdpa`` elsewhere."""
    kind = device_type(device) if device is not None else default_device_type()
    if kind == "cuda":
        return "flash_attention_2"
    logger.warning("flash_attention_2 is CUDA only, using sdpa attention on %s.", kind)
    return "sdpa"
