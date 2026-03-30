# QUEST Workflow

This repo includes a QUEST-oriented workflow for running Forge RL in Northwestern's cluster environment.

## Goals

- Keep QUEST support in-repo so it tracks ongoing Java/RL work.
- Make the main automation path headless and Linux-safe.
- Keep QUEST-specific details in wrapper scripts and docs rather than hard-coding them into model logic.

## Git Strategy

- Work from your existing fork.
- Create QUEST support as feature branches off your active RL branch.
- Merge QUEST portability improvements back into your main RL line so future framework changes inherit them.
- Avoid maintaining a permanent `quest-only` branch or a separate fork unless the codebase intentionally diverges.

## Key Environment Variables

- `FORGE_RL_DATA_DIR`: trajectory JSONL output directory
- `FORGE_RL_PREPROCESSED_DIR`: mmap/preprocessed dataset directory
- `FORGE_RL_CHECKPOINT_DIR`: model checkpoint directory
- `FORGE_RL_LOG_DIR`: log directory
- `FORGE_RL_PPO_TRAJ_DIR`: PPO collection trajectory directory
- `FORGE_RL_EVAL_DIR`: PPO evaluation trajectory directory
- `FORGE_RL_DEVICE`: `cpu` or `cuda`
- `FORGE_JAR_PATH`: optional explicit path to the built desktop fat jar

If these are unset, the repo defaults to local `rl_data/...` paths.

## Recommended QUEST Flow

1. Clone the repo into your QUEST project or home directory.
2. Set the QUEST defaults in `scripts/quest/quest_env.sh` via environment variables:
   - `QUEST_ACCOUNT`
   - `QUEST_CPU_PARTITION`
   - `QUEST_GPU_PARTITION`
   - `QUEST_JAVA_MODULE`
   - `QUEST_PYTHON_MODULE`
   - `QUEST_CUDA_MODULE`
   - optionally `QUEST_MAMBA_MODULE`
   - optionally `QUEST_PYTHON_ENV_TYPE`
   - optionally `QUEST_MAMBA_ENV`
   - optionally `QUEST_MAMBA_INIT`
3. Create the Python environment.

For the repo-managed `venv` workflow:

```bash
bash scripts/quest/setup_env.sh
```

For a `mamba`-managed workflow:

```bash
export QUEST_PYTHON_ENV_TYPE=mamba
export QUEST_MAMBA_ENV=forge_rl
bash scripts/quest/setup_mamba_env.sh
```

If QUEST requires explicit initialization for `mamba`/`conda`, also set:

```bash
export QUEST_MAMBA_INIT="$HOME/.bashrc"
```

4. Build the Java jar:

```bash
bash scripts/01_build.sh
```

5. Submit the workflow stage you need:

```bash
bash scripts/quest/submit_build.sh
bash scripts/quest/submit_collect_preprocess.sh
bash scripts/quest/submit_train_value.sh
bash scripts/quest/submit_train_decisions.sh
bash scripts/quest/submit_ppo.sh
bash scripts/quest/submit_eval.sh
```

## Execution Notes

- Treat `sbatch` as the primary execution path.
- Use interactive jobs only for smoke tests and environment debugging.
- The Java subprocesses and Python model server are designed to run together inside one allocated node using localhost networking.
- The Tk dashboards remain available for workstation use, but the standard QUEST path is now headless.
- The QUEST batch scripts now activate either:
  - the repo `venv` when `QUEST_PYTHON_ENV_TYPE=venv`
  - a named `mamba`/`conda` environment when `QUEST_PYTHON_ENV_TYPE=mamba`

## Fresh-Clone Validation Checklist

- Java 17 and Maven build successfully.
- Python environment installs `requirements.txt`.
- If using `mamba`, the job shell can activate `QUEST_MAMBA_ENV`.
- `bash scripts/01_build.sh` produces the Forge fat jar or `FORGE_JAR_PATH` is set.
- `bash scripts/02_collect_data.sh 10 --clean` runs on a CPU node.
- `bash scripts/03_train_value.sh 1 64 cuda` runs on a GPU node.
- `bash scripts/05_eval.sh "" 4 cpu` completes on a CPU node.
