from contextlib import nullcontext
from dataclasses import dataclass
import platform

import torch


@dataclass(frozen=True)
class RuntimeBackend:
    name: str
    torch_device: object
    display_name: str
    is_gpu: bool
    is_cuda: bool = False
    is_dml: bool = False
    use_amp: bool = False
    pin_memory: bool = False


def _cuda_backend() -> RuntimeBackend | None:
    if not torch.cuda.is_available():
        return None
    return RuntimeBackend(
        name='cuda',
        torch_device='cuda',
        display_name=torch.cuda.get_device_name(0),
        is_gpu=True,
        is_cuda=True,
        use_amp=True,
        pin_memory=True,
    )


def _directml_backend() -> RuntimeBackend | None:
    try:
        import torch_directml
    except ImportError:
        return None

    try:
        device = torch_directml.device()
    except Exception:
        return None

    return RuntimeBackend(
        name='dml',
        torch_device=device,
        display_name='DirectML',
        is_gpu=True,
        is_dml=True,
        use_amp=False,
        pin_memory=False,
    )


def _cpu_backend() -> RuntimeBackend:
    return RuntimeBackend(
        name='cpu',
        torch_device='cpu',
        display_name='CPU',
        is_gpu=False,
    )


def resolve_backend(requested: str | None = None) -> RuntimeBackend:
    requested_name = (requested or '').strip().lower()

    if requested_name in ('cuda', 'gpu'):
        backend = _cuda_backend()
        if backend is None:
            raise RuntimeError('CUDA was requested but is not available.')
        return backend

    if requested_name in ('dml', 'directml'):
        backend = _directml_backend()
        if backend is None:
            raise RuntimeError(
                'DirectML was requested but torch-directml is not installed or no DirectML device is available.'
            )
        return backend

    if requested_name == 'cpu':
        return _cpu_backend()

    cuda_backend = _cuda_backend()
    if cuda_backend is not None:
        return cuda_backend

    if platform.system().lower() == 'windows':
        dml_backend = _directml_backend()
        if dml_backend is not None:
            return dml_backend

    return _cpu_backend()


def auto_device_name() -> str:
    return resolve_backend().name


def autocast_context(backend: RuntimeBackend):
    if backend.use_amp:
        return torch.amp.autocast('cuda', enabled=True)
    return nullcontext()


def create_grad_scaler(backend: RuntimeBackend):
    if backend.use_amp:
        return torch.amp.GradScaler('cuda')
    return None


def supports_cuda_telemetry(backend: RuntimeBackend) -> bool:
    return backend.is_cuda


def is_cuda_device(device) -> bool:
    if isinstance(device, str):
        return device.startswith('cuda')
    return str(device).startswith('cuda')
