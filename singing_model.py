"""Load the pinned upstream singer without initializing unused random weights.

The architecture and checkpoint remain upstream SoulX-Singer. This is a loading
optimization, not a new model or a clean-room implementation. Keep nonpersistent
rotary/DSP buffers materialized; a blanket meta-device context loses them.
"""
from pathlib import Path


def load_singer(checkpoint_path, config, device='mps'):
    import torch
    from accelerate import init_empty_weights
    from soulxsinger.models.soulxsinger import SoulXSinger
    with init_empty_weights(include_buffers=False):
        model = SoulXSinger(config)
    checkpoint = torch.load(Path(checkpoint_path), map_location='cpu',
                            weights_only=True, mmap=True)
    model.load_state_dict(checkpoint['state_dict'], strict=True, assign=True)
    remaining = [name for name, value in
                 list(model.named_parameters())+list(model.named_buffers()) if value.is_meta]
    if remaining:
        raise RuntimeError('Singer tensors were not loaded: '+', '.join(remaining))
    model.eval().to(device)
    if str(device).startswith('mps'):
        torch.mps.synchronize()
    return model
