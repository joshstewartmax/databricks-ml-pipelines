sudo apt-get install openjdk-17-jdk

polars can read from delta, but can only write to parquet directly
only spark can write delta



# Databricks ML Pipelines

This repository contains a simple machine learning pipeline organised for
use with Databricks Lakeflow.  The pipeline is implemented as a Python
package and managed with [uv](https://docs.astral.sh/uv/).

## Workflows

* **Prod pipeline** – defined as a Databricks asset bundle in
  `databricks/prod/bundle.yml`. Each pipeline step is a separate task
  executed from the project wheel with its own cluster configuration.
* **Dev pipeline** – a flexible job definition in `databricks/dev/job.yml`
  that developers can modify and run from branches.  It uses a distinct
  MLflow experiment name.
* **Local pipeline** – run all steps sequentially on a small stub
  dataset to verify code before pushing changes.

## Running locally

```bash
uv run run-pipeline
```

The command above executes all pipeline steps with a local MLflow
tracking directory (`./mlruns`).  Each step logs metrics and artifacts in
its own nested MLflow run.

## Configuration

Configuration is managed with [Hydra](https://hydra.cc/).  Default
settings live under `src/ml_pipelines/conf`.  To override the pipeline
profile used by MLflow:

```bash
uv run run-pipeline pipeline=databricks-dev
```

## Steps

1. **prepare_data** – loads data (stubbed locally) and performs a
   train/test split.
2. **train** – runs a small hyperparameter search and trains a random
   forest model.  The model is logged to MLflow and a registration is
   attempted.
3. **evaluate** – evaluates the trained model on the test split and logs
   metrics and a ROC curve.
4. **feature_importance** – computes permutation feature importances and
   logs the results to MLflow.
5. **model_qa** – placeholder for comparing the new model with the
   currently deployed model.

Placeholders are left for environment‑specific settings such as cluster
configuration, experiment paths and model registry names.
