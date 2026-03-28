import json
import unittest
from pathlib import Path
import shutil

import numpy as np

import sys
import types

ROOT = Path(__file__).resolve().parents[2] / "main" / "python"
sys.path.insert(0, str(ROOT))

# Stub heavy training dependencies so we can import the PPO loader without a
# full torch runtime in the test environment. load_ppo_data itself only uses
# numpy/json/pathlib.
torch_mod = types.ModuleType("torch")
torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
torch_mod.Tensor = object
sys.modules.setdefault("torch", torch_mod)
sys.modules.setdefault("torch.nn", types.ModuleType("torch.nn"))
sys.modules.setdefault("torch.nn.functional", types.ModuleType("torch.nn.functional"))
sys.modules.setdefault("torch.optim", types.ModuleType("torch.optim"))
torch_utils_mod = types.ModuleType("torch.utils")
torch_utils_data_mod = types.ModuleType("torch.utils.data")
torch_utils_data_mod.Dataset = object
torch_utils_mod.data = torch_utils_data_mod
sys.modules.setdefault("torch.utils", torch_utils_mod)
sys.modules.setdefault("torch.utils.data", torch_utils_data_mod)

model_pkg = types.ModuleType("model")
sys.modules.setdefault("model", model_pkg)

backend_mod = types.ModuleType("model.backend")
backend_mod.resolve_backend = lambda *args, **kwargs: None
sys.modules.setdefault("model.backend", backend_mod)

gpu_mod = types.ModuleType("model.gpu_config")
gpu_mod.auto_detect_profile = lambda *args, **kwargs: None
sys.modules.setdefault("model.gpu_config", gpu_mod)

mtg_mod = types.ModuleType("model.mtg_model")
mtg_mod.MTGModel = object
sys.modules.setdefault("model.mtg_model", mtg_mod)

serving_pkg = types.ModuleType("serving")
sys.modules.setdefault("serving", serving_pkg)
model_server_mod = types.ModuleType("serving.model_server")
model_server_mod.ModelServer = object
sys.modules.setdefault("serving.model_server", model_server_mod)

from training.ppo_trainer import load_ppo_data


class CounterfactualLeakageTest(unittest.TestCase):
    def setUp(self):
        self.work_root = Path(__file__).resolve().parent / "_tmp_counterfactual"
        if self.work_root.exists():
            shutil.rmtree(self.work_root)
        self.work_root.mkdir(parents=True)

    def tearDown(self):
        if self.work_root.exists():
            shutil.rmtree(self.work_root)

    def _write_traj(self, directory: Path, with_counterfactual: bool):
        header = {
            "gameId": "g1",
            "won": True,
            "totalDecisions": 1,
            "durationMs": 10,
        }
        record = {
            "turnIndex": 0,
            "decisionType": "DECLARE_ATTACKERS",
            "contextInfo": "declare_attackers",
            "globalFeatures": [0.0] * 64,
            "gameStateFlat": [0.0] * 37216,
            "candidateFeatures": [[0.0] * 256, [0.1] * 256],
            "selectedIndices": [1],
            "actionProbabilities": [0.2, 0.8],
            "valueEstimate": 0.25,
        }
        if with_counterfactual:
            record.update({
                "source": "ppo",
                "modelSelectedIndices": [1],
                "counterfactualHeuristicSelectedIndices": [0],
                "counterfactualHeuristicAvailable": True,
                "counterfactualLabelSource": "shadow_heuristic",
            })
        path = directory / ("traj_augmented.jsonl" if with_counterfactual else "traj_plain.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
            f.write(json.dumps(record) + "\n")

    def test_ppo_loader_ignores_counterfactual_metadata(self):
        dir_plain = self.work_root / "plain"
        dir_aug = self.work_root / "aug"
        dir_plain.mkdir()
        dir_aug.mkdir()
        self._write_traj(dir_plain, with_counterfactual=False)
        self._write_traj(dir_aug, with_counterfactual=True)

        plain = load_ppo_data(dir_plain)
        aug = load_ppo_data(dir_aug)

        self.assertEqual(len(plain[0]), 1)
        self.assertEqual(len(aug[0]), 1)

        plain_attack = plain[0][0]
        aug_attack = aug[0][0]

        self.assertTrue(np.array_equal(
            plain_attack["action_mask"], aug_attack["action_mask"]))
        self.assertTrue(np.array_equal(
            plain_attack["creature_features"], aug_attack["creature_features"]))
        self.assertEqual(
            plain_attack["old_log_prob"], aug_attack["old_log_prob"])
        self.assertEqual(
            plain_attack["advantage"], aug_attack["advantage"])


if __name__ == "__main__":
    unittest.main()
