continue simplifying mlflow_utils
- set_experiment_from_cfg isn't necessary
- try/catches aren't necessary on most of these (just don't run them unless we're on databricks)
- do we need to get the parent run id or can we just let nested=True do it?
- create_parent_run doesn't need all the experiment checking stuff
- terminate_run is unnecessary
- download_artifact is unnecessary

init_mlflow_experiment_and_run_config is identical across pipeline steps

enable running the pipeline locally

lock prod pipeline for editing; try to make it only deployable from main.

can we get rid of initialize and finalize pipeline functions? 
