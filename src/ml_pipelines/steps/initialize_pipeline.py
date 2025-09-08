from __future__ import annotations

import os

from omegaconf import DictConfig
import hydra

from ml_pipelines.mlflow_utils import (
    set_experiment_from_cfg,
    create_parent_run,
    set_task_value,
    get_databricks_run_identifiers,
    generate_pipeline_run_id,
)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    exp_used = set_experiment_from_cfg(cfg)
    print(f"[initialize] Experiment selected: {exp_used}")

    identifiers = get_databricks_run_identifiers()
    pipeline_run_id = generate_pipeline_run_id()

    # Derive tags similar to pipeline.py
    tags = {
        "pipeline_run_id": pipeline_run_id,
        "git_sha": os.environ.get("GIT_SHA", "<GIT_SHA_PLACEHOLDER>"),
        "pipeline_version": os.environ.get("PIPELINE_VERSION", "0.1.0"),
        "orchestrator_run_id": identifiers.get("job_run_id") or "databricks_job",
        "data_version": os.environ.get("DATA_VERSION", "stub"),
    }

    parent_run_id = create_parent_run(cfg, tags=tags)
    print(f"[initialize] Created parent run_id: {parent_run_id}")

    # Persist for downstream tasks
    set_task_value("parent_run_id", parent_run_id)
    set_task_value("pipeline_run_id", pipeline_run_id)
    print("[initialize] Stored task values: parent_run_id and pipeline_run_id")

    # Also expose via environment for non-Databricks debugging
    os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run_id
    os.environ["PIPELINE_RUN_ID"] = pipeline_run_id

    # Do not activate the run context here; keep it open via API until finalize


if __name__ == "__main__":  # pragma: no cover
    main()


