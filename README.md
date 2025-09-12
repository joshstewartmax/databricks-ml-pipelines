## Databricks ML Pipelines – Lakeflow + Hydra + MLflow

This repo demonstrates a maintainable alternative to ad‑hoc Databricks notebooks for ML pipelines. It shows how to:
- **Break the pipeline into steps** with clear interfaces and different compute profiles
- **Run steps as Databricks Lakeflow Job tasks** using Python wheel entrypoints
- **Configure behavior with Hydra** (local/dev/prod overlays)
- **Track experiments and artifacts with MLflow**, including dataset lineage for Delta inputs
- **Use Spark only for data preparation**, persist intermediate datasets as Delta in a Volume, then **switch to Polars** for fast, lightweight model workflows

### Why this over notebooks?
- **Data source logging**: Delta inputs can be logged as MLflow Datasets with helpful tags
- **Dataset artifacts managed**: Deterministic Delta paths per run/step stored in Task Values
- **Composable compute**: Spark cluster for `prepare_data`, single large-memory node for CPU-bound steps
- **Consistent experiment naming** via Hydra config
- **Maintainable code**: Regular Python modules, typed utilities, testable step functions
- **Faster iteration**: Polars replaces Pandas for big speed-ups on single-node steps

---

## High-level architecture

```mermaid
flowchart LR
  subgraph Lakeflow Job (Databricks)
    A[prepare_data\nSpark Cluster] --> B[train\nSingle Node]
    B --> C[evaluate\nSingle Node]
    B --> D[feature_importance\nSingle Node]
    C & D --> E[model_qa\nSingle Node]
  end

  subgraph Storage
    V[/Databricks Volume\n/Volumes/ml_artifacts/<env>/<volume>/<experiment>/.../]
  end

  A -- writes Delta --> V
  B -- reads Delta via Polars --> V
  C -- reads Delta via Polars --> V
  D -- reads Delta via Polars --> V
```

- Each box corresponds to a Python entrypoint wired as a Databricks task.
- `prepare_data` runs on Spark, writes train/test Delta tables to a deterministic path within a Databricks Volume.
- All downstream steps use Polars to read Delta directly via `scan_delta` (no Spark session overhead).

---

## Orchestration with Databricks Asset Bundles (Lakeflow)

- Bundle definition: `databricks.yml`
  - Defines bundle name, targets (`dev`, `prod`), and the wheel artifact build via `uv build --wheel`.
  - Exposes a variable `pipeline` used to select the Hydra profile (e.g., `databricks-dev`, `databricks-prod`).
- Job definition: `resources/ml_pipeline.job.yml`
  - Tasks run Python wheel entrypoints from `pyproject.toml [project.scripts]`:
    - `prepare-data`, `train-model`, `evaluate-model`, `feature-importance`, `model-qa`
  - `depends_on` wires task order: `prepare_data -> train -> (evaluate & feature_importance) -> model_qa`
  - All tasks currently point to an `existing_cluster_id`. In practice, you would configure a Spark cluster for `prepare_data` and a single-node CPU instance for others.

```12:58:/home/josh/repos/databricks-ml-pipelines/resources/ml_pipeline.job.yml
resources:
  jobs:
    ml_pipeline_dev:
      name: ml-pipeline-dev
      ...
      tasks:
        - task_key: prepare_data
          python_wheel_task:
            package_name: databricks-ml-pipelines
            entry_point: prepare-data
            parameters: ["pipeline=${var.pipeline}"]
          libraries:
            - whl: ../dist/*.whl
          existing_cluster_id: "0609-090345-4rjuomtp"
        - task_key: train
          depends_on:
            - task_key: prepare_data
          python_wheel_task:
            package_name: databricks-ml-pipelines
            entry_point: train-model
            parameters: ["pipeline=${var.pipeline}"]
          libraries:
            - whl: ../dist/*.whl
          existing_cluster_id: "0609-090345-4rjuomtp"
        ...
```

- Hydra profile selection is passed via `parameters: ["pipeline=${var.pipeline}"]`. The bundle target sets this variable:

```21:52:/home/josh/repos/databricks-ml-pipelines/databricks.yml
bundle:
  name: databricks_ml_pipelines
...
variables:
  pipeline:
    description: Hydra pipeline config to select (local/databricks-dev/databricks-prod)
    default: databricks-dev
...
targets:
  dev:
    mode: development
    variables:
      pipeline: databricks-dev
  prod:
    mode: production
    variables:
      pipeline: databricks-prod
```

---

## Configuration with Hydra

Hydra organizes configuration under `src/ml_pipelines/conf`.

- Base config: `conf/config.yaml`
  - Declares `defaults: - pipeline: local`, seeds, data locations, and step I/O contracts (keys stored/retrieved via Task Values).
  - Each step has a `step_name` that becomes the MLflow run name.

```1:20:/home/josh/repos/databricks-ml-pipelines/src/ml_pipelines/conf/config.yaml
defaults:
  - pipeline: local
  - _self_

seed: 42

data:
  delta_prefix:
    local: ".data/delta"
    databricks: "/Volumes/ml_artifacts"
  volume_name: "testing"
```

- Environment overlays:
  - `conf/pipeline/local.yaml` → local runs, `env_name: local`, `experiment_name: local_pipeline_experiment`
  - `conf/pipeline/databricks-dev.yaml` → dev workspace experiment `/Shared/ml-pipelines-dev`
  - `conf/pipeline/databricks-prod.yaml` → prod workspace experiment `/Shared/ml-pipelines-prod`

Entry points resolve config via Hydra:
- Local full pipeline: `ml_pipelines/local_pipeline.py` uses `initialize/compose` with `overrides=["pipeline=local"]` to run all steps sequentially in one process, using an in-memory Task Values store.
- Databricks tasks: each step has `@hydra.main(config_path="../conf", config_name="config")` and reads the selected profile from the `pipeline` CLI arg provided as a job parameter.

---

## MLflow integration

- Parent pipeline run is created explicitly so all step runs can be nested under it:
  - `util/mlflow.begin_pipeline_run` creates a run in the configured experiment and returns the run_id.
  - Local orchestrator wraps all steps in `with mlflow.start_run(run_id=pipeline_run_id)`.
  - The final `model_qa` step calls `end_pipeline_run(..., status="FINISHED")` on Databricks.
- Step wrapper `runner.run_step` handles:
  - Opening a nested MLflow run with `run_name=step_cfg.step_name`
  - Setting tag `step=<step_key>`
  - Logging the step’s resolved Hydra config as `config.json`
  - Recording the step run_id to Task Values
- Dataset lineage: `util/mlflow.log_delta_input` optionally logs Delta inputs as MLflow Datasets with useful tags (`dataset_path`, `dataset_version`). Steps `train` and `feature_importance` call this when `mlflow.log_datasets: true`.

```14:54:/home/josh/repos/databricks-ml-pipelines/src/ml_pipelines/runner.py
def run_step(...):
    with mlflow.start_run(run_name=step_cfg.step_name, nested=parent_run_id is not None, parent_run_id=parent_run_id):
        mlflow.set_tag("step", step_key)
        mlflow.log_dict(step_cfg_dict, "config.json")
        result = step_func(cfg, task_values, **step_inputs)
        task_values.set(key=f"{step_key}_run_id", value=current_run.info.run_id)
```

---

## Data strategy: Spark → Delta → Polars

- `prepare_data` is Spark-only and writes Delta tables:
  - Synthetic data created in Spark, split into train/test
  - Written to Delta with deterministic paths computed by `util/delta_paths.build_delta_path`
  - Paths are stored in Task Values and used by downstream steps

```14:36:/home/josh/repos/databricks-ml-pipelines/src/ml_pipelines/steps/prepare_data.py
train_uri = build_delta_path(cfg, "prepare_data", "train")
...
train_df.write.format("delta").mode("overwrite").option("delta.enableDeletionVectors", "false").save(train_uri)
```

- Downstream steps use Polars to read Delta directly, avoiding Spark overhead:

```17:23:/home/josh/repos/databricks-ml-pipelines/src/ml_pipelines/steps/train.py
train_pl = pl.scan_delta(train_uri).collect()
X_pl = train_pl.drop("label")
...
```

- Delta path format is stable and discoverable:
  - Local: `${data.delta_prefix.local}/${env}/${experiment}/pipeline_run_<id>/<step_name>_<run_id8>/<dataset_name>`
  - Databricks: `${data.delta_prefix.databricks}/${env}/${volume}/${experiment}/pipeline_run_<id>/<step_name>_<run_id8>/<dataset_name>`

```9:58:/home/josh/repos/databricks-ml-pipelines/src/ml_pipelines/util/delta_paths.py
# resolves env, experiment, parent run id, and step run id → path
```

---

## Step interfaces and Task Values

To pass metadata between Databricks tasks, the repo abstracts Databricks Task Values:
- `util/task_values.py` provides `DatabricksTaskValues` (backed by `dbutils.jobs.taskValues`) and `LocalTaskValues` (in-memory for local runs).
- Step I/O contracts are declared in Hydra under `steps.<name>.inputs/outputs`:
  - `prepare_data` outputs `train_uri`, `test_uri`, `pipeline_run_id`
  - `train` reads `train_uri`, writes `train_run_id`
  - `evaluate` reads `test_uri` and `train_run_id`, writes `test_auc`
  - `feature_importance` reads `train_uri` and `train_run_id`, writes a boolean flag
  - `model_qa` writes `qa_complete`

```36:54:/home/josh/repos/databricks-ml-pipelines/src/ml_pipelines/util/task_values.py
class DatabricksTaskValues(TaskValues):
    def set(...): self._dbutils.jobs.taskValues.set(...)
    def get(...): return self._dbutils.jobs.taskValues.get(...)
```

---

## Entry points and running

- Local full pipeline (single process):
  - Build venv + install deps
  - Run: `python -m ml_pipelines.local_pipeline` or `uv run local-pipeline`
- Databricks Lakeflow Job:
  1. Build wheel: `uv build --wheel`
  2. Deploy bundle: `databricks bundle deploy -t dev`
  3. Run job: `databricks bundle run ml_pipeline_dev -t dev`

Entrypoints from `pyproject.toml`:
- `local-pipeline` → `ml_pipelines.local_pipeline:main`
- `prepare-data` → `ml_pipelines.steps.prepare_data:main`
- `train-model` → `ml_pipelines.steps.train:main`
- `evaluate-model` → `ml_pipelines.steps.evaluate:main`
- `feature-importance` → `ml_pipelines.steps.feature_importance:main`
- `model-qa` → `ml_pipelines.steps.model_qa:main`

---

## Experiment tracking conventions

- Experiments are set via Hydra profiles:
  - Local: `local_pipeline_experiment`
  - Dev: `/Shared/ml-pipelines-dev`
  - Prod: `/Shared/ml-pipelines-prod`
- Run naming:
  - Parent run: `pipeline_run`
  - Step runs: `01_prepare_data`, `02_train`, `03_evaluate`, `04_feature_importance`, `05_model_qa`
- Tags and artifacts:
  - Tag `step=<step_key>` on each step run
  - `config.json` contains the resolved step config for reproducibility
  - Optional dataset inputs logged for Delta sources (`mlflow.log_datasets: true`)

---

## Notes and extensions
- Swap the `existing_cluster_id` per task to match your desired compute profile (Spark vs single-node).
- Replace synthetic data with your Spark-based feature engineering in `prepare_data`.
- Add real QA checks in `model_qa` and wire promotion logic to MLflow Model Registry if desired.
- If using Unity Catalog tables instead of Volume paths, `log_delta_input` supports `table_name` and `version`.

