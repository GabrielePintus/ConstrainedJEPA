# ConstrainedJEPA

ConstrainedJEPA is an experimental workspace for studying **test-time
constraints in JEPA-based model predictive planning**.

The starting point is
[lucas-maes/le-wm](https://github.com/lucas-maes/le-wm), the official code for
LeWorldModel (LeWM). LeWM learns a pixel-based joint-embedding predictive world
model and uses planning-time optimization over action sequences to reach a goal
observation. This repository is meant to build on top of that setup, using
pretrained checkpoints and datasets when useful, while keeping our own
constraint experiments separate from the upstream implementation.

## Research Question

Can we impose useful constraints at test time during JEPA planning optimization
without retraining the world model?

The main idea is to keep the pretrained JEPA frozen and modify the planning
objective or candidate-action update rule. In LeWM-style planning, the planner
optimizes candidate action sequences by calling a world-model cost function.
That gives us a natural place to add constraints:

```text
total_cost =
    goal_reaching_cost
    + lambda_constraint * constraint_violation
```

The constraints may depend on actions, predicted latent rollouts, observations,
environment state, or external safety/task rules.

## Planned Repository Structure

```text
.
├── configs/
│   ├── experiments/          # Hydra/CLI experiment configs for constrained runs
│   └── constraints/          # Constraint definitions and sweepable parameters
├── docs/                     # Notes, design docs, and experiment writeups
├── notebooks/                # Interactive planning experiments
├── scripts/
│   ├── prepare_checkpoints.py # Helpers for checkpoint conversion/downloads
│   └── run_eval.py           # Entrypoints for constrained planning evaluations
├── src/constrained_jepa/
│   ├── constraints/          # Constraint functions and violation metrics
│   ├── planning/             # Cost wrappers, solver adapters, projections
│   ├── metrics/              # Evaluation and constraint-satisfaction metrics
│   └── utils/                # Shared loading, paths, logging helpers
├── tests/                    # Unit tests for constraints and wrappers
├── pyproject.toml            # uv-managed project metadata
└── README.md
```

This repository should not vendor large upstream repositories, datasets, or
checkpoints. Upstream code can be cloned elsewhere, installed as a dependency,
or referenced by path during development. Large artifacts should live under
`$STABLEWM_HOME`, which is also what `stable-worldmodel` and LeWM expect.

## Constraint Directions

Initial experiments should stay close to the planning API:

1. **Action penalties**
   Penalize control magnitude, jerk, nonsmooth actions, action bounds, or
   forbidden action regions.

2. **Hard candidate filtering**
   For sampling-based optimizers such as CEM or MPPI, assign infeasible
   candidates a very large cost or exclude them before elite selection.

3. **Latent trajectory constraints**
   Use the JEPA rollout embeddings and penalize predicted latent trajectories
   that enter unsafe, implausible, or task-invalid regions.

4. **State/proprioceptive constraints**
   When evaluation info includes state or proprioception, use that signal to
   define interpretable constraints or train a lightweight probe from latent
   embeddings to physical quantities.

5. **Projection or repair**
   Instead of only penalizing candidates, project candidate action sequences
   back into the feasible set before evaluating them.

## Setup With uv

Create and activate a local virtual environment:

```bash
uv venv --python=3.10
source .venv/bin/activate
```

Install the project in editable mode:

```bash
uv pip install -e .
```

Install LeWM/runtime dependencies as needed:

```bash
uv pip install -e ".[lewm]"
```

For notebooks, also install the notebook kernel extra:

```bash
uv pip install -e ".[notebooks]"
```

The `lewm` extra intentionally avoids environment/baseline extras that pull old
`gym==0.21` dependencies. It is enough for checkpoint conversion and cost-model
loading.

For LeWM checkpoints and datasets, use a separate artifact directory:

```bash
export STABLEWM_HOME="$PWD/artifacts/stablewm"
```

The upstream LeWM README explains how checkpoint names map to
`$STABLEWM_HOME`, including the `_object.ckpt` format expected by
`stable_worldmodel.policy.AutoCostModel`.

Recommended local layout:

```text
artifacts/stablewm/
├── pusht/
│   └── lewm_object.ckpt
├── tworoom/
│   └── lewm_object.ckpt
├── cube/
│   └── lewm_object.ckpt
└── reacher/
    └── lewm_object.ckpt
```

With that layout, the checkpoint names passed to LeWM/stable-worldmodel are:

```text
pusht/lewm
tworoom/lewm
cube/lewm
reacher/lewm
```

## Downloading Checkpoints

The Hugging Face mirrors provide `weights.pt` and `config.json` for each
environment:

```bash
uv pip install huggingface-hub

hf download quentinll/lewm-pusht --local-dir "$STABLEWM_HOME/hf/pusht"
hf download quentinll/lewm-tworooms --local-dir "$STABLEWM_HOME/hf/tworoom"
hf download quentinll/lewm-cube --local-dir "$STABLEWM_HOME/hf/cube"
hf download quentinll/lewm-reacher --local-dir "$STABLEWM_HOME/hf/reacher"
```

Those files then need to be converted once into the `_object.ckpt` format that
`stable_worldmodel.policy.AutoCostModel` loads:

```bash
uv pip install -e ".[lewm]"
python scripts/convert_lewm_checkpoints.py
```

After conversion, a model can be loaded by its run name:

```python
import stable_worldmodel as swm

model = swm.policy.AutoCostModel("pusht/lewm", cache_dir="artifacts/stablewm")
```

## Development Principles

- Keep pretrained models frozen unless an experiment explicitly studies
  fine-tuning.
- Put test-time interventions behind small wrappers or adapters instead of
  editing upstream LeWM directly.
- Track both task performance and constraint satisfaction.
- Prefer simple baselines first: unconstrained planning, soft penalties, hard
  filtering, then projection/repair.
- Keep datasets, checkpoints, rendered videos, and experiment outputs out of
  git.

## Near-Term Milestones

1. Define a minimal constrained-cost wrapper around a LeWM-compatible
   `get_cost(info_dict, action_candidates)` model.
2. Implement action-only constraints that do not require decoding latent
   predictions.
3. Reproduce one upstream evaluation with an unconstrained pretrained checkpoint.
4. Add constrained CEM experiments and compare success rate, cost, violation
   rate, and planning time.
5. Add latent or state-based constraints once the baseline loop is reliable.
