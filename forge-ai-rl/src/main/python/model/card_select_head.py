"""
Card Selection Head — General-purpose selector for choosing cards from a set.
Used for discard, sacrifice, scry, and other "choose N cards" effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .game_state_encoder import _make_transformer_encoder


class CardSelectHead(nn.Module):
    """
    Selects N cards from a candidate set. Uses the same pointer network
    approach as TargetHead but specialized for card selection with
    context about why the selection is happening.
    """

    def __init__(self, state_dim: int = 512, card_feature_dim: int = 256,
                 hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.card_projection = nn.Linear(card_feature_dim, hidden_dim)
        self.state_projection = nn.Linear(state_dim, hidden_dim)

        # Self-attention among candidates
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.candidate_attention = _make_transformer_encoder(encoder_layer, 1)

        # Score each card for selection
        self.score_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # GRU for sequential selection (multi-card)
        self.selection_gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, game_state: torch.Tensor, card_features: torch.Tensor,
                card_mask: torch.Tensor) -> torch.Tensor:
        """Single-pass scoring of all candidates."""
        cards = self.card_projection(card_features)
        state = self.state_projection(game_state)

        # Self-attention among candidates
        cards = self.candidate_attention(cards, src_key_padding_mask=~card_mask)

        # Combine with game state
        state_expanded = state.unsqueeze(1).expand(-1, cards.shape[1], -1)
        combined = torch.cat([cards, state_expanded], dim=-1)

        logits = self.score_network(combined).squeeze(-1)
        logits = logits.masked_fill(~card_mask, float('-inf'))
        return logits

    def select_cards(self, game_state: torch.Tensor, card_features: torch.Tensor,
                     card_mask: torch.Tensor, num_select: int) -> tuple:
        """
        Select exactly num_select cards. Uses autoregressive selection.
        Returns (selected_indices, total_log_prob).
        """
        cards = self.card_projection(card_features)
        state = self.state_projection(game_state)
        cards = self.candidate_attention(cards, src_key_padding_mask=~card_mask)

        selected = []
        total_log_prob = torch.zeros(game_state.shape[0], device=game_state.device)
        current_mask = card_mask.clone()
        context = state

        for _ in range(num_select):
            if not current_mask.any():
                break

            state_exp = context.unsqueeze(1).expand(-1, cards.shape[1], -1)
            combined = torch.cat([cards, state_exp], dim=-1)
            logits = self.score_network(combined).squeeze(-1)
            logits = logits.masked_fill(~current_mask, float('-inf'))

            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            total_log_prob += dist.log_prob(action)
            selected.append(action)

            # Update context
            selected_card = cards[torch.arange(cards.size(0)), action]
            context = self.selection_gru(selected_card, context)

            # Remove from candidates
            current_mask = current_mask.scatter(1, action.unsqueeze(1), False)

        if selected:
            return torch.stack(selected, dim=1), total_log_prob
        return torch.zeros(game_state.shape[0], 0, dtype=torch.long, device=game_state.device), total_log_prob
