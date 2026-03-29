#!/usr/bin/env python3
"""
Export trained MTGModel to ONNX format for Java inference.

Produces 9 separate ONNX files:
  state_encoder.onnx   — game state → 512-dim embedding
  value_head.onnx      — embedding → win probability
  priority_head.onnx   — embedding + actions → spell logits
  target_head.onnx     — embedding + targets → target logits
  attack_head.onnx     — embedding + creatures → attack logits
  block_head.onnx      — embedding + blockers/attackers → assignment logits
  card_select_head.onnx — embedding + cards → selection logits
  mulligan_head.onnx   — embedding + hand → keep logit
  binary_head.onnx     — embedding → yes/no logit

Usage:
    python tools/export_onnx.py --checkpoint /path/to/model.pt --output-dir /path/to/onnx/
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.mtg_model import MTGModel


# Fixed sizes for ONNX (must match Java inference)
MAX_BOARD = 40
MAX_HAND = 15
MAX_GY = 20
MAX_STACK = 10
MAX_ACTIONS = 50
MAX_BLOCKERS = 40
MAX_ATTACKERS = 40
GLOBAL_DIM = 96
CARD_DIM = 256
ACTION_DIM = 64
STATE_DIM = 512
OPSET = 18


class StateEncoderWrapper(nn.Module):
    """Wraps the state encoder for ONNX export.
    Patches CardSetEncoder.forward to remove data-dependent control flow
    (the 'if not any_valid.any()' branch) that ONNX can't handle."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # Patch all CardSetEncoders to use ONNX-safe forward
        self._patch_card_set_encoders()

    def _patch_card_set_encoders(self):
        """Replace forward method on CardSetEncoders to remove branches."""
        from model.game_state_encoder import CardSetEncoder
        for name, module in self.encoder.named_modules():
            if isinstance(module, CardSetEncoder):
                module.forward = self._make_safe_forward(module)

    @staticmethod
    def _make_safe_forward(cse):
        """Create a branch-free forward for CardSetEncoder.
        Uses only tensor ops — no if/clone/boolean indexing."""
        def safe_forward(card_features, mask):
            x = cse.card_projection(card_features)
            # Ensure at least one mask entry is True per batch item
            # by ORing the first position with True.
            # Use boolean ops here because ONNX Runtime rejects Max(bool, bool).
            first_col = torch.ones_like(mask[:, :1], dtype=torch.bool)
            safe_mask = torch.cat([
                torch.logical_or(mask[:, :1], first_col),
                mask[:, 1:]
            ], dim=1)
            attn_mask = ~safe_mask
            x = cse.transformer(x, src_key_padding_mask=attn_mask)
            x = cse.output_norm(x)
            # Pool using original mask (not safe_mask)
            mask_expanded = mask.unsqueeze(-1).float()
            counts = mask_expanded.sum(dim=1).clamp(min=1)
            pooled = (x * mask_expanded).sum(dim=1) / counts
            return pooled
        return safe_forward

    def forward(self, global_features,
                my_board, my_board_mask,
                opp_board, opp_board_mask,
                hand, hand_mask,
                my_gy, my_gy_mask,
                opp_gy, opp_gy_mask,
                stack, stack_mask):
        return self.encoder(
            global_features,
            my_board, my_board_mask.bool(),
            opp_board, opp_board_mask.bool(),
            hand, hand_mask.bool(),
            my_gy, my_gy_mask.bool(),
            opp_gy, opp_gy_mask.bool(),
            stack, stack_mask.bool())


class HeadWrapper(nn.Module):
    """Wraps a head that takes (state, features, mask)."""
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, state, features, mask):
        return self.head(state, features, mask.bool())


class BlockHeadWrapper(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, state, blocker_features, blocker_mask,
                attacker_features, attacker_mask):
        return self.head(state,
                         blocker_features, blocker_mask.bool(),
                         attacker_features, attacker_mask.bool())


class ValueWrapper(nn.Module):
    def __init__(self, value_net):
        super().__init__()
        self.value_net = value_net

    def forward(self, state):
        return self.value_net(state)


class BinaryWrapper(nn.Module):
    def __init__(self, binary_head):
        super().__init__()
        self.binary_head = binary_head

    def forward(self, state):
        return self.binary_head(state)


class MulliganWrapper(nn.Module):
    def __init__(self, mulligan_head):
        super().__init__()
        self.mulligan_head = mulligan_head

    def forward(self, state, hand_features, hand_mask):
        # Returns just the keep logit (scalar)
        return self.mulligan_head(state, hand_features, hand_mask.bool())


def export_model(checkpoint_path, output_dir, device='cpu'):
    print(f"Loading model from {checkpoint_path}...", flush=True)
    model = MTGModel.load(checkpoint_path, device=device)
    model.eval()
    print(f"Model loaded. Config: {model.config}", flush=True)

    os.makedirs(output_dir, exist_ok=True)

    # Common dummy inputs
    batch = 1

    def dummy_state_inputs():
        return (
            torch.randn(batch, GLOBAL_DIM, device=device),
            torch.randn(batch, MAX_BOARD, CARD_DIM, device=device),
            torch.ones(batch, MAX_BOARD, device=device),
            torch.randn(batch, MAX_BOARD, CARD_DIM, device=device),
            torch.ones(batch, MAX_BOARD, device=device),
            torch.randn(batch, MAX_HAND, CARD_DIM, device=device),
            torch.ones(batch, MAX_HAND, device=device),
            torch.randn(batch, MAX_GY, CARD_DIM, device=device),
            torch.ones(batch, MAX_GY, device=device),
            torch.randn(batch, MAX_GY, CARD_DIM, device=device),
            torch.ones(batch, MAX_GY, device=device),
            torch.randn(batch, MAX_STACK, CARD_DIM, device=device),
            torch.ones(batch, MAX_STACK, device=device),
        )

    # 1. State Encoder
    print("Exporting state_encoder.onnx...", flush=True)
    encoder = StateEncoderWrapper(model.state_encoder)
    encoder.eval()
    inputs = dummy_state_inputs()
    torch.onnx.export(
        encoder, inputs,
        os.path.join(output_dir, "state_encoder.onnx"),
        opset_version=OPSET,
        input_names=['global_features',
                     'my_board', 'my_board_mask',
                     'opp_board', 'opp_board_mask',
                     'hand', 'hand_mask',
                     'my_gy', 'my_gy_mask',
                     'opp_gy', 'opp_gy_mask',
                     'stack', 'stack_mask'],
        output_names=['state_embedding'],
        do_constant_folding=True,
    )
    # Get reference state embedding for verification
    with torch.no_grad():
        ref_state = encoder(*inputs)
    print(f"  state_embedding shape: {ref_state.shape}", flush=True)

    # 2. Value Head
    print("Exporting value_head.onnx...", flush=True)
    value_wrapper = ValueWrapper(model.value_network)
    value_wrapper.eval()
    torch.onnx.export(
        value_wrapper,
        (ref_state,),
        os.path.join(output_dir, "value_head.onnx"),
        opset_version=OPSET,
        input_names=['state_embedding'],
        output_names=['value'],
        do_constant_folding=True,
    )

    # 3. Priority Head
    print("Exporting priority_head.onnx...", flush=True)
    n_actions = MAX_ACTIONS  # 50
    pri_wrapper = HeadWrapper(model.priority_head)
    pri_wrapper.eval()
    torch.onnx.export(
        pri_wrapper,
        (ref_state,
         torch.randn(batch, n_actions, ACTION_DIM, device=device),
         torch.ones(batch, n_actions, device=device)),
        os.path.join(output_dir, "priority_head.onnx"),
        opset_version=OPSET,
        input_names=['state_embedding', 'action_features', 'action_mask'],
        output_names=['logits'],
        do_constant_folding=True,
    )

    # 4. Target Head
    print("Exporting target_head.onnx...", flush=True)
    n_targets = MAX_BOARD  # 40
    tgt_wrapper = HeadWrapper(model.target_head)
    tgt_wrapper.eval()
    torch.onnx.export(
        tgt_wrapper,
        (ref_state,
         torch.randn(batch, n_targets, CARD_DIM, device=device),
         torch.ones(batch, n_targets, device=device)),
        os.path.join(output_dir, "target_head.onnx"),
        opset_version=OPSET,
        input_names=['state_embedding', 'target_features', 'target_mask'],
        output_names=['logits'],
        do_constant_folding=True,
    )

    # 5. Attack Head
    print("Exporting attack_head.onnx...", flush=True)
    n_creatures = MAX_BOARD  # 40
    atk_wrapper = HeadWrapper(model.attack_head)
    atk_wrapper.eval()
    torch.onnx.export(
        atk_wrapper,
        (ref_state,
         torch.randn(batch, n_creatures, CARD_DIM, device=device),
         torch.ones(batch, n_creatures, device=device)),
        os.path.join(output_dir, "attack_head.onnx"),
        opset_version=OPSET,
        input_names=['state_embedding', 'creature_features', 'creature_mask'],
        output_names=['logits'],
        do_constant_folding=True,
    )

    # 6. Block Head
    print("Exporting block_head.onnx...", flush=True)
    n_blockers = MAX_BLOCKERS  # 40
    n_attackers = MAX_ATTACKERS  # 40
    blk_wrapper = BlockHeadWrapper(model.block_head)
    blk_wrapper.eval()
    torch.onnx.export(
        blk_wrapper,
        (ref_state,
         torch.randn(batch, n_blockers, CARD_DIM, device=device),
         torch.ones(batch, n_blockers, device=device),
         torch.randn(batch, n_attackers, CARD_DIM, device=device),
         torch.ones(batch, n_attackers, device=device)),
        os.path.join(output_dir, "block_head.onnx"),
        opset_version=OPSET,
        input_names=['state_embedding',
                     'blocker_features', 'blocker_mask',
                     'attacker_features', 'attacker_mask'],
        output_names=['logits'],
        do_constant_folding=True,
    )

    # 7. Card Select Head
    print("Exporting card_select_head.onnx...", flush=True)
    n_cards = MAX_BOARD  # 40
    cs_wrapper = HeadWrapper(model.card_select_head)
    cs_wrapper.eval()
    torch.onnx.export(
        cs_wrapper,
        (ref_state,
         torch.randn(batch, n_cards, CARD_DIM, device=device),
         torch.ones(batch, n_cards, device=device)),
        os.path.join(output_dir, "card_select_head.onnx"),
        opset_version=OPSET,
        input_names=['state_embedding', 'card_features', 'card_mask'],
        output_names=['logits'],
        do_constant_folding=True,
    )

    # 8. Mulligan Head
    print("Exporting mulligan_head.onnx...", flush=True)
    n_hand = 7
    mul_wrapper = MulliganWrapper(model.mulligan_head)
    mul_wrapper.eval()
    torch.onnx.export(
        mul_wrapper,
        (ref_state,
         torch.randn(batch, n_hand, CARD_DIM, device=device),
         torch.ones(batch, n_hand, device=device)),
        os.path.join(output_dir, "mulligan_head.onnx"),
        opset_version=OPSET,
        input_names=['state_embedding', 'hand_features', 'hand_mask'],
        output_names=['keep_logit'],
        do_constant_folding=True,
    )

    # 9. Binary Head
    print("Exporting binary_head.onnx...", flush=True)
    bin_wrapper = BinaryWrapper(model.binary_head)
    bin_wrapper.eval()
    torch.onnx.export(
        bin_wrapper,
        (ref_state,),
        os.path.join(output_dir, "binary_head.onnx"),
        opset_version=OPSET,
        input_names=['state_embedding'],
        output_names=['logit'],
        do_constant_folding=True,
    )

    # Verify all exports
    print("\nVerifying ONNX exports...", flush=True)
    import onnxruntime as ort

    files = [f for f in os.listdir(output_dir) if f.endswith('.onnx')]
    total_size = 0
    for f in sorted(files):
        path = os.path.join(output_dir, f)
        size = os.path.getsize(path)
        total_size += size
        # Quick load test
        sess = ort.InferenceSession(path)
        inputs = [i.name for i in sess.get_inputs()]
        outputs = [o.name for o in sess.get_outputs()]
        print(f"  {f}: {size/1024:.0f}KB, inputs={inputs}, outputs={outputs}")
        del sess

    print(f"\nTotal: {len(files)} ONNX files, {total_size/1024/1024:.1f}MB")
    print("Export complete!", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Export MTGModel to ONNX')
    parser.add_argument('--checkpoint',
        default='../../rl_data/checkpoints/model_with_decisions.pt')
    parser.add_argument('--output-dir',
        default='../../rl_data/models')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    export_model(args.checkpoint, args.output_dir, args.device)


if __name__ == '__main__':
    main()
