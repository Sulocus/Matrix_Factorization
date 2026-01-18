"""
Device detection and configuration.
"""

from dataclasses import dataclass
import torch


@dataclass
class DeviceInfo:
    """Information about the compute device."""
    device_type: str
    available_memory_gb: float
    device_name: str
    supports_bf16: bool = False
    supports_tf32: bool = False


def get_device() -> torch.device:
    """Auto-detect and return the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_device_info(device: torch.device = None) -> DeviceInfo:
    """Get detailed information about the device."""
    if device is None:
        device = get_device()

    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        return DeviceInfo(
            device_type='cuda',
            available_memory_gb=props.total_memory / (1024**3),
            device_name=props.name,
            supports_bf16=True,
            supports_tf32=True,
        )
    elif device.type == 'mps':
        return DeviceInfo(
            device_type='mps',
            available_memory_gb=32.0,  # Approximate for Apple Silicon
            device_name='Apple Silicon',
            supports_bf16=False,
            supports_tf32=False,
        )
    else:
        return DeviceInfo(
            device_type='cpu',
            available_memory_gb=64.0,
            device_name='CPU',
            supports_bf16=False,
            supports_tf32=False,
        )


def setup_device(device: torch.device = None) -> tuple[torch.device, DeviceInfo]:
    """
    Setup device with optimal settings.

    Returns:
        device: The torch device
        info: DeviceInfo with device details
    """
    if device is None:
        device = get_device()

    info = get_device_info(device)

    # Enable TF32 for faster matmul on CUDA
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return device, info


def get_compute_dtype(device: torch.device) -> torch.dtype:
    """Get the optimal compute dtype for the device."""
    if device.type == 'cuda':
        return torch.bfloat16
    return torch.float32
