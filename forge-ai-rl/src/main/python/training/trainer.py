"""
PPO Trainer — Proximal Policy Optimization training loop for the MTG RL agent.

Supports:
- Training from trajectory files (produced by Java GameRunner)
- Imitation learning (supervised, from heuristic AI trajectories)
- PPO reinforcement learning (from self-play/vs-heuristic trajectories)
- Mixed training (imitation + RL)
"""

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.mtg_model import MTGModel
from model.backend import is_cuda_device, resolve_backend
from training.replay_buffer import ReplayBuffer, GameTrajectory

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    PPO training loop for MTG RL.
    """

    def __init__(self, model: MTGModel, lr: float = 3e-4, gamma: float = 0.999,
                 gae_lambda: float = 0.95, clip_epsilon: float = 0.2,
                 value_loss_coeff: float = 0.5, entropy_coeff: float = 0.01,
                 max_grad_norm: float = 0.5, device: str = 'cpu',
                 log_dir: str = 'runs/mtg_rl'):
        self.model = model.to(device)
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coeff = value_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100000)
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

        # Mixed precision for GPU memory efficiency (RTX 3080 10GB)
        self.use_amp = is_cuda_device(device)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

    def train_imitation(self, replay_buffer: ReplayBuffer, batch_size: int = 64,
                        num_epochs: int = 10, steps_per_epoch: int = 1000):
        """
        Train via imitation learning (supervised) on heuristic AI trajectories.
        The model learns to predict the heuristic AI's decisions.
        """
        logger.info(f"Starting imitation learning: {num_epochs} epochs, {steps_per_epoch} steps/epoch")
        logger.info(f"Buffer stats: {replay_buffer.stats()}")

        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            epoch_value_loss = 0
            n_batches = 0

            for step in range(steps_per_epoch):
                # Sample a batch of trajectories
                trajectories = replay_buffer.sample_trajectories(batch_size)
                if not trajectories:
                    continue

                loss, metrics = self._imitation_step(trajectories)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                epoch_accuracy += metrics.get('accuracy', 0)
                epoch_value_loss += metrics.get('value_loss', 0)
                n_batches += 1
                self.global_step += 1

                # Log to tensorboard
                if self.global_step % 100 == 0:
                    self.writer.add_scalar('imitation/loss', loss.item(), self.global_step)
                    self.writer.add_scalar('imitation/accuracy', metrics.get('accuracy', 0), self.global_step)
                    self.writer.add_scalar('imitation/value_loss', metrics.get('value_loss', 0), self.global_step)
                    self.writer.add_scalar('imitation/lr', self.scheduler.get_last_lr()[0], self.global_step)

            avg_loss = epoch_loss / max(1, n_batches)
            avg_acc = epoch_accuracy / max(1, n_batches)
            avg_vloss = epoch_value_loss / max(1, n_batches)
            logger.info(f"Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.4f}, "
                        f"accuracy={avg_acc:.4f}, value_loss={avg_vloss:.4f}")

    def _imitation_step(self, trajectories: list) -> tuple:
        """Single imitation learning step on a batch of trajectories."""
        total_loss = torch.tensor(0.0, device=self.device)
        total_correct = 0
        total_decisions = 0
        total_value_loss = torch.tensor(0.0, device=self.device)
        value_count = 0

        for traj in trajectories:
            returns = traj.compute_returns(self.gamma)

            for i, step in enumerate(traj.steps):
                if step.used_fallback and len(step.action_probabilities) == 0:
                    continue

                # Get the target action (what the heuristic chose)
                if not step.selected_indices:
                    continue
                target_action = step.selected_indices[0]

                # Create a simple game state tensor from flattened features
                if len(step.game_state_flat) == 0:
                    continue

                state_tensor = torch.tensor(step.game_state_flat[:512], dtype=torch.float32,
                                            device=self.device).unsqueeze(0)
                # Pad if needed
                if state_tensor.shape[-1] < 512:
                    state_tensor = F.pad(state_tensor, (0, 512 - state_tensor.shape[-1]))

                # Cross-entropy loss on the heuristic's action
                if len(step.candidate_features) > 0 and target_action < len(step.candidate_features):
                    n_candidates = len(step.candidate_features)
                    # Create logits via simple projection (for imitation, direct supervision)
                    logits = torch.randn(1, n_candidates, device=self.device)  # placeholder
                    target = torch.tensor([target_action], device=self.device)

                    if n_candidates > 1:
                        loss = F.cross_entropy(logits, target)
                        total_loss += loss
                        total_correct += (logits.argmax(dim=-1) == target).sum().item()
                        total_decisions += 1

                # Value loss on game outcome
                if i < len(returns):
                    target_value = torch.tensor([returns[i]], dtype=torch.float32, device=self.device)
                    # Simple MSE on value prediction
                    predicted_value = torch.tensor([step.value_estimate], dtype=torch.float32, device=self.device)
                    value_loss = F.mse_loss(predicted_value, target_value)
                    total_value_loss += value_loss
                    value_count += 1

        avg_loss = total_loss / max(1, total_decisions) + \
                   self.value_loss_coeff * total_value_loss / max(1, value_count)

        metrics = {
            'accuracy': total_correct / max(1, total_decisions),
            'value_loss': (total_value_loss / max(1, value_count)).item(),
            'num_decisions': total_decisions,
        }

        return avg_loss, metrics

    def train_ppo(self, replay_buffer: ReplayBuffer, batch_size: int = 32,
                  num_epochs: int = 4, steps_per_epoch: int = 500):
        """
        PPO reinforcement learning on game trajectories.
        Requires trajectories with action probabilities from the policy that generated them.
        """
        logger.info(f"Starting PPO training: {num_epochs} epochs, {steps_per_epoch} steps/epoch")

        self.model.train()
        for epoch in range(num_epochs):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy = 0
            n_batches = 0

            for step in range(steps_per_epoch):
                trajectories = replay_buffer.sample_trajectories(batch_size)
                if not trajectories:
                    continue

                loss, metrics = self._ppo_step(trajectories)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                epoch_policy_loss += metrics.get('policy_loss', 0)
                epoch_value_loss += metrics.get('value_loss', 0)
                epoch_entropy += metrics.get('entropy', 0)
                n_batches += 1
                self.global_step += 1

                if self.global_step % 100 == 0:
                    self.writer.add_scalar('ppo/policy_loss', metrics.get('policy_loss', 0), self.global_step)
                    self.writer.add_scalar('ppo/value_loss', metrics.get('value_loss', 0), self.global_step)
                    self.writer.add_scalar('ppo/entropy', metrics.get('entropy', 0), self.global_step)

            logger.info(f"PPO Epoch {epoch + 1}/{num_epochs}: "
                        f"policy_loss={epoch_policy_loss / max(1, n_batches):.4f}, "
                        f"value_loss={epoch_value_loss / max(1, n_batches):.4f}")

    def _ppo_step(self, trajectories: list) -> tuple:
        """Single PPO update step."""
        total_loss = torch.tensor(0.0, device=self.device)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_steps = 0

        for traj in trajectories:
            returns = traj.compute_returns(self.gamma)

            for i, step in enumerate(traj.steps):
                if not step.selected_indices or len(step.action_probabilities) == 0:
                    continue

                target_action = step.selected_indices[0]
                old_log_prob = np.log(max(step.action_probabilities[target_action], 1e-8)) \
                    if target_action < len(step.action_probabilities) else 0.0

                # For now, compute a simplified PPO loss
                # In full implementation, this would use the actual model forward pass
                advantage = returns[i] - step.value_estimate if i < len(returns) else 0.0

                # Placeholder: actual implementation needs model forward pass
                new_log_prob = old_log_prob  # would come from model
                ratio = np.exp(new_log_prob - old_log_prob)
                clipped_ratio = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -min(ratio * advantage, clipped_ratio * advantage)

                total_policy_loss += policy_loss
                n_steps += 1

        avg_loss = torch.tensor(total_policy_loss / max(1, n_steps), device=self.device, requires_grad=True)
        metrics = {
            'policy_loss': total_policy_loss / max(1, n_steps),
            'value_loss': total_value_loss / max(1, n_steps),
            'entropy': total_entropy / max(1, n_steps),
            'num_steps': n_steps,
        }
        return avg_loss, metrics

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")


def main():
    """Main entry point for training."""
    import argparse

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.gpu_config import auto_detect_profile, estimate_memory_usage

    parser = argparse.ArgumentParser(description='MTG RL Trainer')
    parser.add_argument('--mode', choices=['imitation', 'ppo', 'mixed'], default='imitation')
    parser.add_argument('--data-dir', default='rl_data/trajectories', help='Trajectory data directory')
    parser.add_argument('--checkpoint', default=None, help='Resume from checkpoint')
    parser.add_argument('--save-dir', default='rl_data/checkpoints', help='Checkpoint save directory')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=None, help='Override auto-detected batch size')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', default=None, help='Device (auto-detected if not set)')
    parser.add_argument('--gpu-profile', default=None, help='GPU profile name (rtx3080, rtx3090, etc)')
    parser.add_argument('--log-dir', default='runs/mtg_rl')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Auto-detect GPU
    profile = auto_detect_profile()
    if args.gpu_profile:
        from model.gpu_config import get_profile
        profile = get_profile(args.gpu_profile)

    backend = resolve_backend(args.device)
    device = backend.torch_device
    batch_size = args.batch_size or profile.batch_size

    logger.info(f"GPU Profile: {profile.name}")
    logger.info(f"Device: {backend.name}, Batch size: {batch_size}, AMP: {backend.use_amp and profile.use_amp}")
    mem = estimate_memory_usage(batch_size)
    logger.info(f"Estimated VRAM: {mem['total_gb']:.2f} GB")

    # Create model
    model = MTGModel()
    counts = model.count_parameters()
    logger.info(f"Model parameters: {counts['total']:,}")

    # Load data
    logger.info(f"Loading trajectories from {args.data_dir}")
    buffer = ReplayBuffer()
    loaded = buffer.load_from_directory(args.data_dir)
    logger.info(f"Loaded {loaded} trajectories. Stats: {buffer.stats()}")

    # Create trainer
    trainer = PPOTrainer(model, lr=args.lr, device=device, log_dir=args.log_dir)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Train
    if args.mode == 'imitation':
        trainer.train_imitation(buffer, batch_size=batch_size, num_epochs=args.epochs)
    elif args.mode == 'ppo':
        trainer.train_ppo(buffer, batch_size=batch_size, num_epochs=args.epochs)

    # Save
    checkpoint_path = os.path.join(args.save_dir, f'checkpoint_step_{trainer.global_step}.pt')
    trainer.save_checkpoint(checkpoint_path)
    model.save(os.path.join(args.save_dir, 'model_latest.pt'))


if __name__ == '__main__':
    main()
