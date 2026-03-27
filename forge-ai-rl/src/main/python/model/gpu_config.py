"""
GPU Configuration — Profiles for different GPU memory budgets.

RTX 3080 (10GB VRAM) profile:
- Default model (~15M params) fits comfortably
- Batch size 64 for training, 1 for inference
- Mixed precision (fp16) for 2x memory savings during training
- Gradient checkpointing for large batch sizes

Memory estimates at default dimensions:
- Model parameters: ~15M * 4 bytes = ~60MB (fp32), ~30MB (fp16)
- Batch of 64 game states: ~64 * (40*256 + 40*256 + 15*256 + 20*256 + 20*256 + 10*256) * 4 ≈ ~100MB
- Gradients + optimizer states: ~180MB (fp32), ~120MB (fp16)
- Total training: ~400-600MB (well within 10GB)
"""

from dataclasses import dataclass

from .backend import resolve_backend


@dataclass
class GPUProfile:
    """GPU-specific training configuration."""
    name: str
    vram_gb: float
    batch_size: int
    accumulation_steps: int  # gradient accumulation for effective larger batch
    use_amp: bool  # automatic mixed precision
    use_gradient_checkpointing: bool
    max_game_state_dim: int
    max_board_size: int
    max_hand_size: int
    num_workers: int  # dataloader workers


# Pre-defined profiles
RTX_3080 = GPUProfile(
    name="RTX 3080 (10GB)",
    vram_gb=10.0,
    batch_size=64,
    accumulation_steps=1,
    use_amp=True,  # fp16 mixed precision saves ~50% VRAM
    use_gradient_checkpointing=False,  # not needed at this model size
    max_game_state_dim=512,
    max_board_size=30,
    max_hand_size=15,
    num_workers=4,
)

RTX_3090 = GPUProfile(
    name="RTX 3090 (24GB)",
    vram_gb=24.0,
    batch_size=128,
    accumulation_steps=1,
    use_amp=True,
    use_gradient_checkpointing=False,
    max_game_state_dim=512,
    max_board_size=30,
    max_hand_size=15,
    num_workers=4,
)

RTX_4090 = GPUProfile(
    name="RTX 4090 (24GB)",
    vram_gb=24.0,
    batch_size=256,
    accumulation_steps=1,
    use_amp=True,
    use_gradient_checkpointing=False,
    max_game_state_dim=768,
    max_board_size=40,
    max_hand_size=20,
    num_workers=8,
)

CPU_ONLY = GPUProfile(
    name="CPU Only",
    vram_gb=0,
    batch_size=16,
    accumulation_steps=4,
    use_amp=False,
    use_gradient_checkpointing=False,
    max_game_state_dim=256,
    max_board_size=20,
    max_hand_size=10,
    num_workers=2,
)

DIRECTML_GPU = GPUProfile(
    name="Windows DirectML GPU",
    vram_gb=16.0,
    batch_size=64,
    accumulation_steps=1,
    use_amp=False,
    use_gradient_checkpointing=False,
    max_game_state_dim=512,
    max_board_size=30,
    max_hand_size=15,
    num_workers=2,
)

PROFILES = {
    'rtx3080': RTX_3080,
    'rtx3090': RTX_3090,
    'rtx4090': RTX_4090,
    'directml': DIRECTML_GPU,
    'dml': DIRECTML_GPU,
    'cpu': CPU_ONLY,
}


def get_profile(name: str = 'rtx3080') -> GPUProfile:
    """Get a GPU profile by name."""
    return PROFILES.get(name.lower(), RTX_3080)


def auto_detect_profile() -> GPUProfile:
    """Auto-detect GPU and return appropriate profile."""
    try:
        backend = resolve_backend()
        if backend.is_dml:
            return DIRECTML_GPU
        if not backend.is_cuda:
            return CPU_ONLY

        import torch

        gpu_name = torch.cuda.get_device_name(0).lower()
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)

        if '4090' in gpu_name:
            return RTX_4090
        elif '3090' in gpu_name or vram_gb > 20:
            return RTX_3090
        elif '3080' in gpu_name or vram_gb >= 9:
            return RTX_3080
        else:
            # Small GPU, use conservative settings
            profile = GPUProfile(**vars(RTX_3080))
            profile.name = f"Auto: {torch.cuda.get_device_name(0)} ({vram_gb:.1f}GB)"
            profile.batch_size = max(8, int(32 * vram_gb / 10))
            return profile

    except ImportError:
        return CPU_ONLY
    except RuntimeError:
        return CPU_ONLY


def estimate_memory_usage(batch_size: int = 64, state_dim: int = 512,
                          board_size: int = 40, card_dim: int = 256) -> dict:
    """Estimate VRAM usage in MB."""
    # Model parameters (rough estimate)
    model_params_mb = 15 * 4  # ~15M params * 4 bytes (fp32)

    # Input tensors per batch
    per_sample_mb = (
        96 * 4 +  # global features
        board_size * card_dim * 4 * 2 +  # my board + opp board
        15 * card_dim * 4 +  # hand
        20 * card_dim * 4 * 2 +  # graveyards
        10 * card_dim * 4  # stack
    ) / (1024 * 1024)
    input_mb = per_sample_mb * batch_size

    # Activations (rough: 3x model size per sample)
    activation_mb = model_params_mb * 3 * batch_size / 64

    # Optimizer states (Adam: 2x model params)
    optimizer_mb = model_params_mb * 2

    # Gradients
    gradient_mb = model_params_mb

    total = model_params_mb + input_mb + activation_mb + optimizer_mb + gradient_mb

    return {
        'model_mb': model_params_mb,
        'input_mb': input_mb,
        'activation_mb': activation_mb,
        'optimizer_mb': optimizer_mb,
        'gradient_mb': gradient_mb,
        'total_mb': total,
        'total_gb': total / 1024,
    }
