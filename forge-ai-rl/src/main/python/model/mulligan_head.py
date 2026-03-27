"""
Mulligan Head — Decides whether to keep an opening hand or mulligan.
Also selects which cards to put back for London mulligan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .game_state_encoder import _make_transformer_encoder


class MulliganHead(nn.Module):
    """
    Two-phase mulligan decision:
    1. Keep or mulligan the current hand
    2. If keeping after mulligan, choose cards to put on bottom
    """

    def __init__(self, state_dim: int = 512, card_feature_dim: int = 256,
                 hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.card_projection = nn.Linear(card_feature_dim, hidden_dim)
        self.state_projection = nn.Linear(state_dim, hidden_dim)

        # Hand evaluation via self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.hand_encoder = _make_transformer_encoder(encoder_layer, 2)

        # Keep/mulligan decision
        self.keep_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Card selection for bottom (same as card_select but with mulligan context)
        self.bottom_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # higher score = more likely to bottom
        )

    def forward(self, game_state: torch.Tensor, hand_features: torch.Tensor,
                hand_mask: torch.Tensor) -> torch.Tensor:
        """Forward for training compatibility.
        Returns keep logit per sample (positive=keep, negative=mull)."""
        keep_logit, _ = self.evaluate_hand(game_state, hand_features, hand_mask)
        return keep_logit.squeeze(-1)

    def evaluate_hand(self, game_state: torch.Tensor, hand_features: torch.Tensor,
                      hand_mask: torch.Tensor) -> tuple:
        """
        Evaluate an opening hand.

        Returns:
            keep_logit: (batch, 1) — positive means keep, negative means mulligan
            card_bottom_scores: (batch, max_hand) — score for putting each card on bottom
        """
        cards = self.card_projection(hand_features)
        state = self.state_projection(game_state)

        # Self-attention among hand cards
        cards_attn = self.hand_encoder(cards, src_key_padding_mask=~hand_mask)

        # Pool hand for keep/mulligan decision
        mask_expanded = hand_mask.unsqueeze(-1).float()
        counts = mask_expanded.sum(dim=1).clamp(min=1)
        hand_pooled = (cards_attn * mask_expanded).sum(dim=1) / counts

        # Keep/mulligan
        combined = torch.cat([hand_pooled, state], dim=-1)
        keep_logit = self.keep_classifier(combined)

        # Bottom scores for each card
        state_expanded = state.unsqueeze(1).expand(-1, cards_attn.shape[1], -1)
        card_combined = torch.cat([cards_attn, state_expanded], dim=-1)
        bottom_scores = self.bottom_scorer(card_combined).squeeze(-1)
        bottom_scores = bottom_scores.masked_fill(~hand_mask, float('-inf'))

        return keep_logit, bottom_scores

    def decide_keep(self, game_state, hand_features, hand_mask) -> tuple:
        """Sample keep/mulligan decision. Returns (keep: bool tensor, log_prob)."""
        keep_logit, _ = self.evaluate_hand(game_state, hand_features, hand_mask)
        prob = torch.sigmoid(keep_logit.squeeze(-1))
        keep = torch.bernoulli(prob).bool()
        log_prob = F.logsigmoid(keep_logit.squeeze(-1)) * keep.float() + \
                   F.logsigmoid(-keep_logit.squeeze(-1)) * (~keep).float()
        return keep, log_prob

    def choose_bottom_cards(self, game_state, hand_features, hand_mask, num_bottom) -> tuple:
        """Choose which cards to put on bottom. Returns (indices, log_prob)."""
        _, bottom_scores = self.evaluate_hand(game_state, hand_features, hand_mask)

        selected = []
        total_log_prob = torch.zeros(game_state.shape[0], device=game_state.device)
        current_mask = hand_mask.clone()

        for _ in range(num_bottom):
            dist = torch.distributions.Categorical(logits=bottom_scores.masked_fill(~current_mask, float('-inf')))
            action = dist.sample()
            total_log_prob += dist.log_prob(action)
            selected.append(action)
            current_mask = current_mask.scatter(1, action.unsqueeze(1), False)

        return torch.stack(selected, dim=1) if selected else torch.empty(0), total_log_prob
