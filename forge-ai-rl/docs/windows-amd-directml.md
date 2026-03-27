# Windows 11 AMD DirectML Setup

This repo can now use DirectML for training and inference on supported AMD GPUs such as the Radeon RX 7800 XT.

## Prerequisites

- Windows 11
- Current AMD Adrenalin driver
- Python 3.11+ virtual environment for `forge-ai-rl`
- Java build of the desktop app with the RL module enabled

## Python Environment

From `forge-ai-rl`:

```bash
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r src/main/python/requirements.txt
```

The Windows requirements include:

- `torch-directml` for PyTorch training on AMD GPUs
- `onnxruntime-directml` for ONNX inference on DirectML

## Training

Examples:

```bash
python src/main/python/training/training_ui.py --device dml
python src/main/python/training/train_decisions_ui.py --device dml
python src/main/python/training/ppo_ui.py --device dml
python src/main/python/training/awr_trainer.py --device dml
```

If DirectML is unavailable, omit `--device` to let the repo fall back automatically.

## Python Inference

```bash
python src/main/python/serving/model_server.py --model rl_data/checkpoints/model_with_decisions.pt --device dml
```

## Java ONNX Inference

The Java ONNX client now supports an execution provider setting:

- `auto`
- `directml`
- `cpu`

On Windows, `auto` will try DirectML first and fall back to CPU if the local ONNX Runtime build does not expose the DirectML provider.
