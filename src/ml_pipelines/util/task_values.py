from __future__ import annotations

from typing import Dict, Optional, Any


class TaskValues:
    """
    Base class for interacting with Databricks task values so we can create a 
    single abstraction for pipeline runs locally and on Databricks.
    """

    def set(self, key: str, value: Any, task_key: Optional[str] = None) -> None:
        raise NotImplementedError

    def get(self, key: str, task_key: Optional[str] = None) -> Optional[Any]:
        raise NotImplementedError


class LocalTaskValues(TaskValues):
    """In-memory implementation suitable for local pipeline runs."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def set(self, key: str, value: Any, task_key: Optional[str] = None) -> None:
        namespace = task_key or "pipeline"
        if namespace not in self._store:
            self._store[namespace] = {}
        self._store[namespace][key] = value

    def get(self, key: str, task_key: Optional[str] = None) -> Optional[Any]:
        namespace = task_key or "pipeline"
        return self._store.get(namespace, {}).get(key)


class DatabricksTaskValues(TaskValues):
    """Databricks implementation backed by dbutils.jobs.taskValues."""

    def __init__(self) -> None:
        # Lazy import so local runs don't require dbutils
        from ml_pipelines.util.databricks import get_dbutils  # type: ignore

        self._dbutils = get_dbutils()

    def set(self, key: str, value: Any, task_key: Optional[str] = None) -> None:
        # task_key is ignored on set: values are written for the current task
        self._dbutils.jobs.taskValues.set(key=key, value=value)

    def get(self, key: str, task_key: Optional[str] = None) -> Optional[Any]:
        if task_key is None:
            # When task_key is absent, attempt to read from current task scope
            return self._dbutils.jobs.taskValues.get(key=key)
        return self._dbutils.jobs.taskValues.get(key=key, taskKey=task_key)


