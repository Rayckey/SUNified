# Architecture and stage handoffs

Import project code through `kuafu` and resolve resources through `kuafu.paths`.
Select the external Isaac Lab runtime with `ISAACLAB_ROOT` or
`[workspace].isaaclab_root`.

```text
agents → asset_generation → isaac
   └────→ programs ↔ mpc → data → bc → rl
              ↑              └──────→ sim2real
            skills

pipeline: run state, stage orchestration and artifact handoffs
```

Agent tools expose asset construction, scene observation, program submission,
and verification. `pipeline` coordinates collection and learning stages.

## SUN Program components

| Component | Responsibility |
|---|---|
| `SUNProgramPlan` | Pending tasks, atomic actions, and completed history |
| `SUNProgramLearner` | Task planning, grounding, and constraint generation |
| `SUNConstraintGenerator` | Symbolic constraint proposals and repairs |
| `SUNProgramExecutor` | Candidate compilation and MPC execution |
| `SUNProgramEvaluator` | Batched predicates and residual rewards |
| `CompiledSUNProgram` | Saved-stage compilation and per-environment progress |
| `SUNProgramStage` | One stage's actions, constraints, residuals, and predicates |

Use `program_runtime` for a shared executor/evaluator handle. The executor owns
`program_learner`; the evaluator owns `compiled_program`. Save stages under
`program/subtask_*/`.

## Artifact flow

```text
program/subtask_*/{task.txt, atomic.txt, constraints.json, ...}
  + success.txt + stats.txt
        ↓
per-program MPC datasets → exact-count HDF5 + SHA-256
        ↓
BC configurations + sweep CSV → checkpoints → evaluated priors
        ↓
params_best.json + weights_best.json + selected prior
        ↓
residual launcher → PPO logs + checkpoint + completion receipt
```

Store run output under `KUAFU_ARTIFACT_ROOT` (default `../KuafuArtifacts`). Keep
bundled programs in `resources/learned_programs/` and experiment inputs in
`resources/experiment_definitions/`. Pipeline state records canonical paths and hashes.

## Stage contracts

| Stage | Required behavior |
|---|---|
| `learn-programs` | Use Panda formulation scenes with zero rewards and suppressed done flags. Let the executor evaluate candidate predicates. Save successful `program/` runs and success metadata. |
| `collect` | Revalidate programs, divide success targets across workers, and verify exact counts before promoting HDF5. Record seeds, logs, identity, and hashes. |
| `train-prior` | Train from compact HDF5 using manifest-derived configurations; publish the resulting priors. |
| `eval-prior` | Evaluate sweep checkpoints and record comparable success rates, assigned tolerances, and selected reward weights. |
| `train-residual` | Select a compatible evaluated prior and weights. Freeze the prior and train a bounded additive PPO correction. |

## Resume and recovery

Inspect `kuafu status` between long jobs. Completed stages are skipped; use
`--force` to rerun one. Reuse validated per-program files and retain failed-worker
logs for diagnosis. Resolve a partial final dataset before resuming collection.

`--prepare-only` records a prepared residual launcher. Execution records completion
after successful training with a readable checkpoint; exceptions record a failed
state and diagnostic message. See [workflows](WORKFLOWS.md) for commands.
