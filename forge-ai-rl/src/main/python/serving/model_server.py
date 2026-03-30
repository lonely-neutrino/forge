"""
Model Server — TCP server that receives inference requests from Java and returns decisions.

Protocol: length-prefixed JSON over TCP
- Client sends: [4 bytes big-endian length][JSON payload]
- Server responds: [4 bytes big-endian length][JSON payload]

This is the bridge between the Java game engine and the Python neural network.
"""

import json
import socket
import struct
import threading
import logging
import sys
import os
import time
import queue
from collections import namedtuple
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.backend import resolve_backend
from model.mtg_model import MTGModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class ModelServer:
    """
    TCP server for serving RL model inference to the Java game engine.
    """

    # Pending request: (request_dict, result_event, result_holder)
    _InferItem = namedtuple('_InferItem', ['request', 'event', 'result'])

    def __init__(self, model: MTGModel, host: str = 'localhost', port: int = 50051,
                 device: str = 'cpu', batch_wait_ms: float = 5.0, max_batch: int = 32,
                 use_argmax: bool = False):
        self.model = model
        self.model.eval()
        self.host = host
        self.port = port
        self.device = device
        self.running = False
        self.request_count = 0
        self.batch_wait_ms = batch_wait_ms
        self.max_batch = max_batch
        self.use_argmax = use_argmax
        # Queue for batched inference
        self._infer_queue = queue.Queue()
        # Limit concurrent client threads to prevent OOM
        self._client_semaphore = threading.Semaphore(64)

    def start(self):
        """Start the server."""
        self.running = True

        # Start batch inference thread
        infer_thread = threading.Thread(
            target=self._batch_inference_loop, daemon=True)
        infer_thread.start()

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(128)
        logger.info(f"Model server listening on {self.host}:{self.port}")

        try:
            while self.running:
                server_socket.settimeout(1.0)
                try:
                    client_socket, addr = server_socket.accept()
                    thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, addr))
                    thread.daemon = True
                    thread.start()
                except socket.timeout:
                    continue
                except OSError:
                    continue
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.running = False
            try:
                server_socket.close()
            except Exception:
                pass

    def _batch_inference_loop(self):
        """Collect requests and process them in batches on GPU."""
        while self.running:
            # Wait for first item
            try:
                first = self._infer_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch = [first]
            # Collect more items within the wait window
            deadline = time.monotonic() + self.batch_wait_ms / 1000.0
            while len(batch) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._infer_queue.get(timeout=remaining)
                    batch.append(item)
                except queue.Empty:
                    break

            # Process each request individually but without lock contention
            # (this thread is the only one doing inference)
            for item in batch:
                try:
                    response = self._process_request_impl(item.request)
                    item.result.append(response)
                except Exception as e:
                    logger.error(f"Batch inference error: {e}")
                    item.result.append({
                        'selectedIndices': [0],
                        'actionProbabilities': [],
                        'valueEstimate': 0.0,
                    })
                item.event.set()

    def _handle_client(self, client_socket: socket.socket, addr):
        """Handle a single client connection."""
        self._client_semaphore.acquire()
        try:
            client_socket.settimeout(600)  # 10 min timeout per client
            while self.running:
                # Read length prefix
                length_bytes = self._recv_exact(client_socket, 4)
                if not length_bytes:
                    break
                length = struct.unpack('>I', length_bytes)[0]
                if length > 10_000_000:  # 10MB sanity limit
                    break

                # Read payload
                payload = self._recv_exact(client_socket, length)
                if not payload:
                    break

                # Parse and process
                try:
                    request = json.loads(payload.decode('utf-8'))
                    self.request_count += 1
                    if request.get('requestType') == 'metadata':
                        response = self._build_metadata()
                    else:
                        response = self._process_request(request)
                except Exception as e:
                    response = {
                        'selectedIndices': [0],
                        'actionProbabilities': [],
                        'valueEstimate': 0.0,
                    }

                # Send response
                response_bytes = json.dumps(response).encode('utf-8')
                client_socket.sendall(struct.pack('>I', len(response_bytes)))
                client_socket.sendall(response_bytes)

        except Exception:
            pass  # Client disconnected or error
        finally:
            self._client_semaphore.release()
            try:
                client_socket.close()
            except Exception:
                pass

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes."""
        data = b''
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _process_request(self, request: dict) -> dict:
        """Submit request to batch inference queue and wait for result."""
        event = threading.Event()
        result = []  # mutable holder
        item = self._InferItem(request=request, event=event, result=result)
        self._infer_queue.put(item)
        event.wait(timeout=30.0)
        if result:
            return result[0]
        logger.error("Inference timeout - no result after 30s")
        return {
            'selectedIndices': [0],
            'actionProbabilities': [],
            'valueEstimate': 0.0,
        }

    def _build_metadata(self) -> dict:
        return {
            'useArgmax': bool(self.use_argmax),
            'modelId': getattr(self.model, '_loaded_from_path', 'in_memory'),
            'backend': str(self.device),
        }

    @torch.no_grad()
    def _process_request_impl(self, request: dict) -> dict:
        decision_type = request.get('decisionType', 'UNKNOWN')

        try:
            # Parse game state features
            state_embedding = self._encode_game_state(request)

            if decision_type == 'PRIORITY_ACTION':
                return self._handle_priority(state_embedding, request)
            elif decision_type == 'TARGET_SELECTION':
                return self._handle_target(state_embedding, request)
            elif decision_type == 'DECLARE_ATTACKERS':
                return self._handle_attack(state_embedding, request)
            elif decision_type == 'DECLARE_BLOCKERS':
                return self._handle_block(state_embedding, request)
            elif decision_type == 'CARD_SELECTION':
                return self._handle_card_select(state_embedding, request)
            elif decision_type == 'MULLIGAN':
                return self._handle_mulligan(state_embedding, request)
            elif decision_type == 'BINARY_CHOICE':
                return self._handle_binary(state_embedding, request)
            else:
                # Default: return first option
                return {'selectedIndices': [0], 'actionProbabilities': [], 'valueEstimate': 0.0}

        except Exception as e:
            logger.error(f"Error processing {decision_type}: {e}")
            return {'selectedIndices': [0], 'actionProbabilities': [], 'valueEstimate': 0.0}

    def _encode_game_state(self, request: dict) -> torch.Tensor:
        """Encode game state features into a state embedding.
        Uses the same parse_game_state() as training to guarantee
        identical tensor layout."""
        import numpy as np
        from training.mmap_dataset import parse_game_state

        gf = request.get('globalFeatures', [])
        flat = request.get('gameStateFlat', [])

        gf_np = np.array(gf, dtype=np.float32)
        flat_np = np.array(flat, dtype=np.float32)
        np.clip(gf_np, -10, 10, out=gf_np)
        np.clip(flat_np, -10, 10, out=flat_np)
        gf_np = np.nan_to_num(gf_np)
        flat_np = np.nan_to_num(flat_np)

        g, zones, masks = parse_game_state(flat_np, gf_np)

        t = lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)

        return self.model.encode_state(
            t(g),
            t(zones['my_board']), t(masks['my_board_mask']),
            t(zones['opp_board']), t(masks['opp_board_mask']),
            t(zones['hand']), t(masks['hand_mask']),
            t(zones['my_gy']), t(masks['my_gy_mask']),
            t(zones['opp_gy']), t(masks['opp_gy_mask']),
            t(zones['stack']), t(masks['stack_mask'])
        )

    def _handle_priority(self, state: torch.Tensor, request: dict) -> dict:
        candidates = request.get('candidateFeatures', [])
        if not candidates:
            return {'selectedIndices': [0], 'actionProbabilities': [], 'valueEstimate': 0.0}

        action_features = self._to_tensor_2d(candidates, len(candidates), 64)
        action_mask = torch.ones(1, len(candidates), dtype=torch.bool, device=self.device)

        logits = self.model.priority_head(state, action_features, action_mask)
        probs = torch.softmax(logits, dim=-1)
        action = probs.argmax(dim=-1).item() if self.use_argmax else torch.multinomial(probs, 1).item()
        value = self.model.get_value(state).item()

        return {
            'selectedIndices': [action],
            'actionProbabilities': probs[0].cpu().numpy().tolist(),
            'valueEstimate': value,
        }

    def _handle_target(self, state: torch.Tensor, request: dict) -> dict:
        candidates = request.get('candidateFeatures', [])
        if not candidates:
            return {'selectedIndices': [0], 'actionProbabilities': [], 'valueEstimate': 0.0}

        card_dim = self.model.config.get('card_feature_dim', 256)
        target_features = self._to_tensor_2d(candidates, len(candidates), card_dim)
        target_mask = torch.ones(1, len(candidates), dtype=torch.bool, device=self.device)

        # Pass spell context if available
        spell_raw = request.get('spellFeatures')
        spell_features = None
        if spell_raw is not None:
            spell_features = torch.tensor(
                [spell_raw], dtype=torch.float32,
                device=self.device)

        logits = self.model.target_head(state, target_features, target_mask,
                                         spell_features=spell_features)
        probs = torch.softmax(logits, dim=-1)

        max_select = request.get('maxSelections', 1)
        if max_select <= 1:
            action = probs.argmax(dim=-1).item() if self.use_argmax else torch.multinomial(probs, 1).item()
            selected = [action]
        else:
            # Select top-k by probability
            _, indices = probs.topk(min(max_select, len(candidates)), dim=-1)
            selected = indices[0].cpu().numpy().tolist()

        value = self.model.get_value(state).item()
        return {
            'selectedIndices': selected,
            'actionProbabilities': probs[0].cpu().numpy().tolist(),
            'valueEstimate': value,
        }

    _debug_count = 0

    def _handle_attack(self, state: torch.Tensor, request: dict) -> dict:
        # Dump first request to file for comparison with training data
        if ModelServer._debug_count < 3:
            ModelServer._debug_count += 1
            import json as jlib
            with open(f'/tmp/server_request_{ModelServer._debug_count}.json', 'w') as df:
                # Save the raw request (excluding huge arrays for readability)
                debug = {
                    'globalFeatures': request.get('globalFeatures', []),
                    'candidateFeatures': request.get('candidateFeatures', []),
                    'myBoardFeatures_count': len(request.get('myBoardFeatures', [])),
                    'myBoardFeatures_nonzero': sum(1 for r in request.get('myBoardFeatures', []) if any(v!=0 for v in r)),
                    'oppBoardFeatures_count': len(request.get('oppBoardFeatures', [])),
                    'oppBoardFeatures_nonzero': sum(1 for r in request.get('oppBoardFeatures', []) if any(v!=0 for v in r)),
                    'myHandFeatures_count': len(request.get('myHandFeatures', [])),
                    'myHandFeatures_nonzero': sum(1 for r in request.get('myHandFeatures', []) if any(v!=0 for v in r)),
                    'myGraveyardFeatures_count': len(request.get('myGraveyardFeatures', [])),
                    'oppGraveyardFeatures_count': len(request.get('oppGraveyardFeatures', [])),
                    'candidate_first': request.get('candidateFeatures', [[]])[0][:20] if request.get('candidateFeatures') else [],
                    'state_norm': state.norm().item(),
                    'state_5': state[0,:5].tolist(),
                }
                jlib.dump(debug, df, indent=2)
            logger.info(f"DEBUG: Saved request to /tmp/server_request_{ModelServer._debug_count}.json")

        candidates = request.get('candidateFeatures', [])
        if not candidates:
            return {'selectedIndices': [], 'actionProbabilities': [], 'valueEstimate': 0.0}

        creature_features = self._to_tensor_2d(candidates, len(candidates), 256)
        creature_mask = torch.ones(1, len(candidates), dtype=torch.bool, device=self.device)

        logits = self.model.attack_head(state, creature_features, creature_mask)
        probs = torch.sigmoid(logits)
        if self.use_argmax:
            decisions = (probs > 0.5).float()
        else:
            decisions = torch.bernoulli(probs.clamp(0, 1))

        selected = [i for i in range(len(candidates)) if decisions[0, i].item() > 0.5]
        value = self.model.get_value(state).item()

        return {
            'selectedIndices': selected,
            'actionProbabilities': probs[0].cpu().numpy().tolist(),
            'valueEstimate': value,
        }

    def _handle_block(self, state: torch.Tensor, request: dict) -> dict:
        """Handle blocking using the proper BlockHead.

        Input: (blocker, attacker) pair candidates (256-dim each)
        + a "no block" zero vector as the last candidate.

        The BlockHead needs separate blocker and attacker tensors.
        We reconstruct these from the pairs, then sample one
        attacker assignment per blocker (or "don't block").
        """
        candidates = request.get('candidateFeatures', [])
        if not candidates or len(candidates) < 2:
            return {'selectedIndices': [],
                    'actionProbabilities': [],
                    'valueEstimate': 0.0}

        # Last candidate is "no block" (zero vector) — skip it
        real_pairs = candidates[:-1]
        n_pairs = len(real_pairs)
        if n_pairs == 0:
            value = self.model.get_value(state).item()
            return {'selectedIndices': [],
                    'actionProbabilities': [],
                    'valueEstimate': value}

        # Infer number of attackers from pair structure:
        # pairs are ordered b0a0, b0a1, ..., b1a0, b1a1, ...
        # First blocker's features are in pairs[0][:card_dim]
        import numpy as np
        card_dim = self.model.config.get('card_feature_dim', 256)
        first_blocker = np.array(real_pairs[0][:card_dim])
        n_attackers = 1
        for j in range(1, n_pairs):
            other = np.array(real_pairs[j][:card_dim])
            if np.allclose(first_blocker, other, atol=0.01):
                n_attackers += 1
            else:
                break
        n_blockers = n_pairs // max(n_attackers, 1)

        if n_blockers == 0 or n_attackers == 0:
            value = self.model.get_value(state).item()
            return {'selectedIndices': [],
                    'actionProbabilities': [],
                    'valueEstimate': value}

        # Extract unique blocker and attacker features
        bf = torch.zeros(1, n_blockers, card_dim,
                         device=self.device)
        bm = torch.ones(1, n_blockers, dtype=torch.bool,
                        device=self.device)
        af = torch.zeros(1, n_attackers, card_dim,
                         device=self.device)
        am = torch.ones(1, n_attackers, dtype=torch.bool,
                        device=self.device)

        for b in range(n_blockers):
            pair_idx = b * n_attackers
            if pair_idx < n_pairs:
                feats = real_pairs[pair_idx]
                bf[0, b, :min(card_dim, len(feats))] = \
                    torch.tensor(feats[:card_dim],
                                 dtype=torch.float32,
                                 device=self.device)
        for a in range(n_attackers):
            if a < n_pairs:
                feats = real_pairs[a]
                af[0, a, :min(card_dim, len(feats)-card_dim)] = \
                    torch.tensor(feats[card_dim:card_dim*2],
                                 dtype=torch.float32,
                                 device=self.device)

        # BlockHead: (batch, n_blockers, n_attackers+1)
        logits = self.model.block_head(
            state, bf, bm, af, am)

        # Sample one assignment per blocker
        selected_pairs = []
        all_probs = torch.zeros(n_pairs + 1,
                                device=self.device)

        for b in range(n_blockers):
            probs_b = torch.softmax(logits[0, b], dim=-1)
            if self.use_argmax:
                action = probs_b.argmax(dim=-1).item()
            else:
                dist = torch.distributions.Categorical(
                    logits=logits[0, b])
                action = dist.sample().item()

            if action < n_attackers:
                # This blocker blocks attacker 'action'
                pair_idx = b * n_attackers + action
                if pair_idx < n_pairs:
                    selected_pairs.append(pair_idx)

            # Store probs for the pairs belonging to
            # this blocker
            for a in range(n_attackers):
                pidx = b * n_attackers + a
                if pidx < n_pairs:
                    all_probs[pidx] = probs_b[a]
            # "no block" prob contributes to last entry
            all_probs[-1] += probs_b[-1] / n_blockers

        value = self.model.get_value(state).item()
        return {
            'selectedIndices': selected_pairs,
            'actionProbabilities':
                all_probs.cpu().numpy().tolist(),
            'valueEstimate': value,
        }

    def _handle_card_select(self, state: torch.Tensor, request: dict) -> dict:
        candidates = request.get('candidateFeatures', [])
        if not candidates:
            return {'selectedIndices': [], 'actionProbabilities': [], 'valueEstimate': 0.0}

        card_features = self._to_tensor_2d(candidates, len(candidates), 256)
        card_mask = torch.ones(1, len(candidates), dtype=torch.bool, device=self.device)

        logits = self.model.card_select_head(state, card_features, card_mask)
        probs = torch.softmax(logits, dim=-1)

        min_select = request.get('minSelections', 1)
        max_select = request.get('maxSelections', 1)
        num_select = max(min_select, min(max_select, len(candidates)))

        if num_select <= 0:
            selected = []
        elif num_select == 1:
            action = probs.argmax(dim=-1).item() if self.use_argmax else torch.multinomial(probs, 1).item()
            selected = [action]
        else:
            _, indices = probs.topk(min(num_select, len(candidates)), dim=-1)
            selected = indices[0].cpu().numpy().tolist()

        value = self.model.get_value(state).item()
        return {
            'selectedIndices': selected,
            'actionProbabilities': probs[0].cpu().numpy().tolist(),
            'valueEstimate': value,
        }

    def _handle_mulligan(self, state: torch.Tensor, request: dict) -> dict:
        candidates = request.get('candidateFeatures', [])
        if not candidates:
            return {'selectedIndices': [1], 'actionProbabilities': [], 'valueEstimate': 0.0}

        hand_features = self._to_tensor_2d(candidates, len(candidates), 256)
        hand_mask = torch.ones(1, len(candidates), dtype=torch.bool, device=self.device)

        keep_logit, _ = self.model.mulligan_head.evaluate_hand(state, hand_features, hand_mask)
        keep_prob = torch.sigmoid(keep_logit).item()
        keep = keep_prob > 0.5

        value = self.model.get_value(state).item()
        return {
            'selectedIndices': [1 if keep else 0],
            'actionProbabilities': [1 - keep_prob, keep_prob],
            'valueEstimate': value,
        }

    def _handle_binary(self, state: torch.Tensor, request: dict) -> dict:
        logit = self.model.binary_head(state)
        prob = torch.sigmoid(logit).item()
        decision = prob > 0.5

        value = self.model.get_value(state).item()
        return {
            'selectedIndices': [1 if decision else 0],
            'actionProbabilities': [1 - prob, prob],
            'valueEstimate': value,
        }

    # Tensor conversion helpers

    def _to_tensor(self, data, target_shape) -> torch.Tensor:
        """Convert a list to a tensor with target shape, padding with zeros."""
        if isinstance(data, list) and len(data) > 0:
            t = torch.tensor(data, dtype=torch.float32, device=self.device)
            if t.dim() == 1:
                t = t.unsqueeze(0)
            # Pad or truncate to target shape
            result = torch.zeros(target_shape, device=self.device)
            min_len = min(t.shape[-1], target_shape[-1])
            result[..., :min_len] = t[..., :min_len]
            return result
        return torch.zeros(target_shape, device=self.device)

    def _to_tensor_2d(self, data, max_items, feature_dim) -> torch.Tensor:
        """Convert a 2D list to a padded tensor."""
        result = torch.zeros(1, max_items, feature_dim, device=self.device)
        if isinstance(data, list):
            for i, row in enumerate(data):
                if i >= max_items:
                    break
                if isinstance(row, list) and len(row) > 0:
                    min_len = min(len(row), feature_dim)
                    result[0, i, :min_len] = torch.tensor(row[:min_len], dtype=torch.float32, device=self.device)
        return result

    def _to_mask(self, data, max_items) -> torch.Tensor:
        """Convert a boolean list to a mask tensor."""
        result = torch.zeros(1, max_items, dtype=torch.bool, device=self.device)
        if isinstance(data, list):
            for i, val in enumerate(data):
                if i >= max_items:
                    break
                result[0, i] = bool(val)
        return result


def main():
    """Main entry point for the model server."""
    import argparse
    parser = argparse.ArgumentParser(description='MTG RL Model Server')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=50051, help='Server port')
    parser.add_argument('--model', default=None, help='Path to saved model weights')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda/dml)')
    parser.add_argument('--argmax', action='store_true',
                        help='Force deterministic argmax action selection')
    args = parser.parse_args()
    backend = resolve_backend(args.device)

    # Create or load model
    if args.model and os.path.exists(args.model):
        logger.info(f"Loading model from {args.model}")
        model = MTGModel.load(args.model, device=backend.name)
        model._loaded_from_path = args.model
    else:
        logger.info("Creating fresh model with random weights")
        model = MTGModel.from_size("xl")
        model.to(backend.torch_device)
        model._loaded_from_path = 'fresh_model'

    # Print parameter counts
    counts = model.count_parameters()
    logger.info(f"Model parameters: {counts}")

    # Start server
    server = ModelServer(model, host=args.host, port=args.port,
                         device=backend.torch_device,
                         use_argmax=args.argmax)
    server.start()


if __name__ == '__main__':
    main()
