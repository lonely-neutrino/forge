"""
MTGModel — Unified model that combines the game state encoder with all decision heads.
This is the single model object used for training and inference.
"""

import torch
import torch.nn as nn
from .backend import resolve_backend
from .game_state_encoder import GameStateTransformer
from .value_network import ValueNetwork
from .priority_head import PriorityHead
from .target_head import TargetHead
from .combat_attack_head import AttackHead
from .combat_block_head import BlockHead
from .card_select_head import CardSelectHead
from .mulligan_head import MulliganHead
from .binary_head import BinaryHead


class MTGModel(nn.Module):
    """
    Complete MTG RL model with shared game state encoder and specialized decision heads.

    Architecture:
        GameStateTransformer (shared encoder)
            ├── ValueNetwork (critic)
            ├── PriorityHead (which spell to play)
            ├── TargetHead (choose targets)
            ├── AttackHead (declare attackers)
            ├── BlockHead (declare blockers)
            ├── CardSelectHead (discard, sacrifice, scry, etc.)
            ├── MulliganHead (keep/mulligan + bottom cards)
            └── BinaryHead (yes/no decisions)
    """

    def __init__(self, global_feature_dim: int = 96, card_feature_dim: int = 256,
                 action_feature_dim: int = 64, state_dim: int = 512,
                 hidden_dim: int = 256, zone_embed_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        # Shared encoder
        self.state_encoder = GameStateTransformer(
            global_feature_dim=global_feature_dim,
            card_feature_dim=card_feature_dim,
            zone_embed_dim=zone_embed_dim,
            output_dim=state_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Value network (critic)
        self.value_network = ValueNetwork(state_dim, hidden_dim, dropout)

        # Decision heads (actors)
        self.priority_head = PriorityHead(state_dim, action_feature_dim, hidden_dim, num_heads, dropout)
        self.target_head = TargetHead(state_dim, card_feature_dim, hidden_dim, dropout)
        self.attack_head = AttackHead(state_dim, card_feature_dim, hidden_dim, num_heads, dropout)
        self.block_head = BlockHead(state_dim, card_feature_dim, hidden_dim, num_heads, dropout)
        self.card_select_head = CardSelectHead(state_dim, card_feature_dim, hidden_dim, num_heads, dropout)
        self.mulligan_head = MulliganHead(state_dim, card_feature_dim, hidden_dim, num_heads, dropout)
        self.binary_head = BinaryHead(state_dim, hidden_dim, dropout)

        # Store dimensions for serialization
        self.config = {
            'global_feature_dim': global_feature_dim,
            'card_feature_dim': card_feature_dim,
            'action_feature_dim': action_feature_dim,
            'state_dim': state_dim,
            'hidden_dim': hidden_dim,
            'zone_embed_dim': zone_embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': dropout,
        }

    def encode_state(self, global_features, my_board, my_board_mask, opp_board, opp_board_mask,
                     hand, hand_mask, my_gy, my_gy_mask, opp_gy, opp_gy_mask, stack, stack_mask):
        """Encode the game state into a dense vector."""
        return self.state_encoder(
            global_features, my_board, my_board_mask, opp_board, opp_board_mask,
            hand, hand_mask, my_gy, my_gy_mask, opp_gy, opp_gy_mask, stack, stack_mask
        )

    def get_value(self, state_embedding):
        """Get the value estimate for a state."""
        return self.value_network(state_embedding)

    def save(self, path: str):
        """Save model weights and config."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'MTGModel':
        """Load model from saved weights."""
        runtime = resolve_backend(device) if isinstance(device, str) else None
        target_device = runtime.torch_device if runtime is not None else device
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(**checkpoint['config'])
        # Filter out keys with size mismatches (e.g. target_head
        # expanded from 64 to 256-dim)
        model_state = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in checkpoint['state_dict'].items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                skipped.append(k)
        if skipped:
            print(f"Skipped weights (shape mismatch): "
                  f"{skipped[:5]}", flush=True)
        model.load_state_dict(filtered, strict=False)
        model.to(target_device)
        return model

    def count_parameters(self) -> dict:
        """Count parameters in each component."""
        counts = {}
        counts['state_encoder'] = sum(p.numel() for p in self.state_encoder.parameters())
        counts['value_network'] = sum(p.numel() for p in self.value_network.parameters())
        counts['priority_head'] = sum(p.numel() for p in self.priority_head.parameters())
        counts['target_head'] = sum(p.numel() for p in self.target_head.parameters())
        counts['attack_head'] = sum(p.numel() for p in self.attack_head.parameters())
        counts['block_head'] = sum(p.numel() for p in self.block_head.parameters())
        counts['card_select_head'] = sum(p.numel() for p in self.card_select_head.parameters())
        counts['mulligan_head'] = sum(p.numel() for p in self.mulligan_head.parameters())
        counts['binary_head'] = sum(p.numel() for p in self.binary_head.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts
