"""
Attack Declaration Head — Decides which creatures to attack with.

This is a joint binary decision over all possible attackers (attack or don't attack each creature),
implemented as a set-to-set model that considers the full context of all creatures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .game_state_encoder import _make_transformer_encoder


class AttackHead(nn.Module):
    """
    For each potential attacker, outputs a probability of attacking.
    Uses self-attention among attackers so the decision for one creature
    can depend on what other creatures are doing (e.g., "alpha strike" vs "hold back blockers").
    """

    def __init__(self, state_dim: int = 512, card_feature_dim: int = 256,
                 hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.card_projection = nn.Linear(card_feature_dim, hidden_dim)
        self.state_projection = nn.Linear(state_dim, hidden_dim)

        # Self-attention among potential attackers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.attacker_attention = _make_transformer_encoder(encoder_layer, 2)

        # Binary classifier per creature: attack or not
        self.attack_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, game_state: torch.Tensor, creature_features: torch.Tensor,
                creature_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            game_state: (batch, state_dim)
            creature_features: (batch, max_creatures, card_feature_dim)
            creature_mask: (batch, max_creatures) — True for valid potential attackers

        Returns:
            attack_logits: (batch, max_creatures) — logit for each creature to attack
        """
        creatures = self.card_projection(creature_features)
        state = self.state_projection(game_state)

        # Self-attention among creatures
        attn_mask = ~creature_mask
        creatures = self.attacker_attention(creatures, src_key_padding_mask=attn_mask)

        # Concatenate game state context with each creature
        state_expanded = state.unsqueeze(1).expand(-1, creatures.shape[1], -1)
        combined = torch.cat([creatures, state_expanded], dim=-1)

        # Binary decision per creature
        logits = self.attack_classifier(combined).squeeze(-1)
        # Clamp logits to prevent sigmoid saturation — keeps gradients
        # and entropy non-zero so PPO can learn
        logits = logits.clamp(-5.0, 5.0)
        logits = logits.masked_fill(~creature_mask, -100.0)

        return logits

    def sample_attacks(self, game_state: torch.Tensor, creature_features: torch.Tensor,
                       creature_mask: torch.Tensor) -> tuple:
        """
        Sample attack decisions. Returns (attack_decisions, log_probs, entropy).
        attack_decisions: (batch, max_creatures) boolean tensor
        """
        logits = self.forward(game_state, creature_features, creature_mask)
        probs = torch.sigmoid(logits)

        # Sample independently per creature
        decisions = torch.bernoulli(probs.clamp(0, 1))
        decisions = decisions * creature_mask.float()  # zero out padding

        # Log probs
        log_probs = F.logsigmoid(logits) * decisions + F.logsigmoid(-logits) * (1 - decisions)
        log_probs = (log_probs * creature_mask.float()).sum(dim=-1)

        # Entropy
        entropy = -(probs * F.logsigmoid(logits) + (1 - probs) * F.logsigmoid(-logits))
        entropy = (entropy * creature_mask.float()).sum(dim=-1)

        return decisions.bool(), log_probs, entropy
