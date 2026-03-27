"""
Game State Encoder — Transformer-based encoder that converts the game state
feature vectors into a dense embedding used by all decision heads.

Uses set-based attention over cards in each zone (battlefield, hand, graveyard, stack)
since the order of cards on the board doesn't matter — only their relationships do.
"""

import torch
import torch.nn as nn
import math


def _make_transformer_encoder(encoder_layer: nn.TransformerEncoderLayer,
                              num_layers: int) -> nn.TransformerEncoder:
    """Disable nested tensor fast-path so DirectML can run TransformerEncoder."""
    try:
        return nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
    except TypeError:
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if hasattr(encoder, "enable_nested_tensor"):
            encoder.enable_nested_tensor = False
        if hasattr(encoder, "use_nested_tensor"):
            encoder.use_nested_tensor = False
        return encoder


class CardSetEncoder(nn.Module):
    """
    Encodes a set of card feature vectors using multi-head self-attention.
    Cards attend to each other to capture board relationships
    (e.g., equipment attached to creatures, synergies between cards).
    """

    def __init__(self, card_feature_dim: int, embed_dim: int, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.card_projection = nn.Linear(card_feature_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = _make_transformer_encoder(encoder_layer, num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, card_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            card_features: (batch, max_cards, card_feature_dim)
            mask: (batch, max_cards) — True for real cards, False for padding

        Returns:
            zone_embedding: (batch, embed_dim) — pooled representation of the card set
        """
        # Project card features to embedding dimension
        x = self.card_projection(card_features)  # (batch, max_cards, embed_dim)

        # Handle empty zones (all mask False) — return zero embedding
        any_valid = mask.any(dim=1)  # (batch,)
        if not any_valid.any():
            return torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)

        # For samples with no valid cards, set at least one mask entry to True
        # to avoid transformer errors, then zero out the result later
        safe_mask = mask.clone()
        safe_mask[~any_valid, 0] = True

        # Create attention mask (True = ignore, for PyTorch convention)
        attn_mask = ~safe_mask

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        x = self.output_norm(x)

        # Pool: average over real cards (masked mean)
        mask_expanded = mask.unsqueeze(-1).float()  # (batch, max_cards, 1)
        counts = mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)
        pooled = (x * mask_expanded).sum(dim=1) / counts  # (batch, embed_dim)

        # Zero out embeddings for completely empty zones
        pooled = pooled * any_valid.unsqueeze(-1).float()

        return pooled


class GameStateTransformer(nn.Module):
    """
    Full game state encoder. Encodes each zone with a CardSetEncoder,
    concatenates with global features, and produces a single game state embedding.
    """

    def __init__(self, global_feature_dim: int = 96, card_feature_dim: int = 256,
                 zone_embed_dim: int = 128, output_dim: int = 512,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.global_feature_dim = global_feature_dim
        self.card_feature_dim = card_feature_dim
        self.output_dim = output_dim

        # One encoder per zone type
        self.my_board_encoder = CardSetEncoder(card_feature_dim, zone_embed_dim, num_heads, num_layers, dropout)
        self.opp_board_encoder = CardSetEncoder(card_feature_dim, zone_embed_dim, num_heads, num_layers, dropout)
        self.hand_encoder = CardSetEncoder(card_feature_dim, zone_embed_dim, num_heads, num_layers, dropout)
        self.my_graveyard_encoder = CardSetEncoder(card_feature_dim, zone_embed_dim, num_heads, num_layers, dropout)
        self.opp_graveyard_encoder = CardSetEncoder(card_feature_dim, zone_embed_dim, num_heads, num_layers, dropout)
        self.stack_encoder = CardSetEncoder(card_feature_dim, zone_embed_dim, num_heads, num_layers, dropout)

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, zone_embed_dim),
            nn.GELU(),
            nn.LayerNorm(zone_embed_dim),
        )

        # Cross-zone attention: let zone embeddings attend to each other
        cross_layer = nn.TransformerEncoderLayer(
            d_model=zone_embed_dim,
            nhead=num_heads,
            dim_feedforward=zone_embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.cross_zone_transformer = _make_transformer_encoder(cross_layer, 1)

        # Final projection to output_dim
        # 7 zone embeddings: global, my_board, opp_board, hand, my_gy, opp_gy, stack
        self.output_projection = nn.Sequential(
            nn.Linear(zone_embed_dim * 7, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, global_features, my_board, my_board_mask, opp_board, opp_board_mask,
                hand, hand_mask, my_gy, my_gy_mask, opp_gy, opp_gy_mask, stack, stack_mask):
        """
        Args:
            global_features: (batch, global_feature_dim)
            my_board: (batch, max_board, card_feature_dim)
            my_board_mask: (batch, max_board)
            ... (same pattern for each zone)

        Returns:
            game_state_embedding: (batch, output_dim)
        """
        # Encode each zone
        global_embed = self.global_encoder(global_features)
        my_board_embed = self.my_board_encoder(my_board, my_board_mask)
        opp_board_embed = self.opp_board_encoder(opp_board, opp_board_mask)
        hand_embed = self.hand_encoder(hand, hand_mask)
        my_gy_embed = self.my_graveyard_encoder(my_gy, my_gy_mask)
        opp_gy_embed = self.opp_graveyard_encoder(opp_gy, opp_gy_mask)
        stack_embed = self.stack_encoder(stack, stack_mask)

        # Stack zone embeddings for cross-zone attention
        zone_embeddings = torch.stack([
            global_embed, my_board_embed, opp_board_embed,
            hand_embed, my_gy_embed, opp_gy_embed, stack_embed
        ], dim=1)  # (batch, 7, zone_embed_dim)

        # Cross-zone attention
        zone_embeddings = self.cross_zone_transformer(zone_embeddings)

        # Flatten and project
        flat = zone_embeddings.reshape(zone_embeddings.shape[0], -1)  # (batch, 7 * zone_embed_dim)
        output = self.output_projection(flat)  # (batch, output_dim)

        return output
