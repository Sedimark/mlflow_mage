"""MlflowSaver class implementation and related functions to model saving."""

from __future__ import annotations

import os
import platform
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Optional
from typing import Any
    
import mlflow
from mlflow.tracking import MlflowClient

from mlflow_mage.tags import Tags


class MlflowSaver:
    def __init__(self, run_name: Optional[str] = None, tags: Optional[dict[str, str]] = None, nested: bool = False) -> None:
        self._client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI"))
        experiment = self._client.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT_NAME"))
        self._expeiment_id = "0" if experiment is None else experiment.experiment_id
        self._tags = Tags(tags)
        self._run_name = run_name
        self._nested = nested

    def __enter__(self):
        self._start_time = time.time()

        self._tags.update({
            "start_time": str(datetime.now().isoformat()),
            "python_version": platform.python_version(),
            "mlflow_version": mlflow.__version__
        })        

        self._run = mlflow.start_run(
            experiment_id=self._expeiment_id,
            run_name=self._run_name,
            nested=self._nested,
            tags=self._tags.tags,
        )

        self._log_system_info()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._run:
            return

        duration = time.time() - self._start_time
        mlflow.log_metric("duration_seconds", duration)
        
        if exc_type is not None:
            mlflow.set_tag("error", str(exc_val))
            mlflow.set_tag("error_type", exc_type.__name__)
            
        mlflow.end_run()

    def _log_system_info(self):
        """Log system information as parameters"""
        system_info = {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
            "machine": platform.machine()
        }
        
        try:
            import psutil
            system_info.update({
                "cpu_count": psutil.cpu_count(),
                "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
            })
        except ImportError:
            pass
            
        try:
            import torch
            system_info.update({
                "cuda_available": torch.cuda.is_available(),
                "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0
            })
        except ImportError:
            pass
        
        mlflow.log_params({f"sys_{k}": v for k, v in system_info.items()})

    def log_params(self, params_dict: dict[str, object]):
        """
        Log multiple parameters from a dictionary

        Parameters:
        -----------
        param_dict: dict[str, object]
            A dictionary of parameters to be logged through MLFlow.

        Return:
        -------
        None
        """
        string_params = {k: str(v) for k, v in params_dict.items()}
        mlflow.log_params(string_params)
    
    def log_param_groups(self, param_groups: dict[str, dict[str, object]]):
        """
        Log parameter groups (e.g., optimizer param groups)        

        Parameters:
        -----------
        param_groups: dict[str, dict[str, object]]
            A dictionary of parameter groups in order to group parametrs based on some topic.

        Return:
        -------
        None
        """
        for group_name, params in param_groups.items():
            prefixed_params = {f"{group_name}_{k}": v for k, v in params.items()}
            self.log_params(prefixed_params)

    def log_metrics(self, metrics_dict: dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics from a dict using a step as optional.
        
        Parameters:
        -----------
        metrics_dict : dict[str, float]
            A dictionary of name of metrics as keys and value of metrics as values of type float.
        step : Optional[int] = None
            Integer representing the step to log the metrics to. This way we can store the same metrics for the same run multiple times.

        Return:
        -------
        None
        """
        for metric_name, metric_value in metrics_dict.items():
            mlflow.log_metric(metric_name, metric_value, step)

    def log_model(
        self,
        model: Any,
        input_example: Any,
        output_example: Any,
        model_name: str = "model",
        framework: str = "auto",
        pip_requirements: Optional[list[str]] = None
    ):
        """
        Log a model based on the current active run.

        Parameters:
        -----------
        model : Any
            The model that will be logged through MLFlow.
        input_example : Any
            An example of the input that is used by the model to make a prediction.
        input_example : Any
            An example of the output that the model is providing when doing the predection.
        model_name : str
            The name of the model.
        framework : str
            The framework behind the model, can be 'auto', 'sklearn', 'pytorch', 'tensorflow', 'xgboost', 'lightgbm'.
        pip_requirements : Optional[list[str]] = None
            Pip requirements that needs to be installed when using this model through MLflow.

        Return:
        --------
        None
        """
        if framework == "auto":
            try:
                import sklearn
                if isinstance(model, sklearn.base.BaseEstimator):
                    framework = "sklearn"
            except ImportError:
                pass
                
            try:
                import torch.nn
                if isinstance(model, torch.nn.Module):
                    framework = "pytorch"
            except ImportError:
                pass
                
            try:
                import tensorflow
                if isinstance(model, tensorflow.keras.Model):
                    framework = "tensorflow"
            except ImportError:
                pass

        signature = mlflow.models.infer_signature(input_example, output_example)

        model_info = None
        if framework == "sklearn":
            model_info = mlflow.sklearn.log_model(model, name=model_name, signature=signature, pip_requirements=pip_requirements)
        elif framework == "pytorch":
            model_info = mlflow.pytorch.log_model(model, name=model_name, signature=signature, pip_requirements=pip_requirements)
        elif framework == "tensorflow":
            model_info = mlflow.tensorflow.log_model(model, name=model_name, signature=signature, pip_requirements=pip_requirements)
        elif framework == "xgboost":
            model_info = mlflow.xgboost.log_model(model, name=model_name, signature=signature, pip_requirements=pip_requirements)
        elif framework == "lightgbm":
            model_info = mlflow.lightgbm.log_model(model, name=model_name, signature=signature, pip_requirements=pip_requirements)
        else:
            model_info = mlflow.pyfunc.log_model(model, name=model_name, signature=signature, pip_requirements=pip_requirements)

        if model_info is not None:
            self._model_uri = model_info.model_uri

    def set_tags(self, tags_dict: dict[str, str]):
        """Set multiple tags from a dictionary"""
        mlflow.set_tags(tags_dict)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a local file or directory as an artifact

        Parameters:
        ----------
        local_path : str 
            String pointing to a local directory where the artifacts are present.
        artifact_path : Optional[str] = None
            The path where the artifacts will be stored inside MinIO.

        Return:
        --------
        None
        """
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log all the files in a directory as artifacts

        Parameters:
        ----------
        local_dir : str 
            String pointing to a local directory where the artifacts are present.
        artifact_path : Optional[str] = None
            The path where the artifacts will be stored inside MinIO.

        Return:
        --------
        None
        """
        mlflow.log_artifacts(local_dir, artifact_path)

    @contextmanager
    def create_child_run(self, run_name: Optional[str] = None, 
                        tags: Optional[dict[str, str]] = None):
        """Create a nested child run"""
        child_tags = tags or {}
        child_run_name = run_name or f"child_{uuid.uuid4().hex[:8]}"
        
        with MlflowSaver(
             run_name=child_run_name, 
             nested=True,
             tags=child_tags
         ) as child:
            yield child

    @property
    def model_uri(self) -> str:
        """
        Return the logged model_uri of the current run.

        Return:
        --------
        str
            The run id of the current run.
        """
        if not self._run:
            raise ValueError("No active run. Make sure to call within a context manager.")

        if not self._model_uri:
            raise ValueError("No model was logged, please make sure to log the model.")

        return self._model_uri

    
    def register_model(
        self, model_uri: str, name: str, 
        description: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        await_registration_for: Optional[int] = None
    ) -> str:
        """
        Register a model in the MLflow Model Registry
        
        Parameters:
        -----------
        model_uri : str
            URI of the model, can be in format 'runs:/<run_id>/<artifact_path>'
        name : str
            Name to register the model under
        description : str, optional
            Description for the model version
        tags : Dict[str, str], optional
            Tags to associate with the model version
        await_registration_for : int, optional
            Seconds to wait for the model version to become READY
            
        Returns:
        --------
        str
            The version of the registered model
        """
        model_details = mlflow.register_model(
            model_uri=model_uri,
            name=name,
            await_registration_for=await_registration_for
        )
        
        if description is not None:
            self._client.update_model_version(
                name=name,
                version=model_details.version,
                description=description
            )
        
        if tags is not None:
            for key, value in tags.items():
                self._client.set_model_version_tag(
                    name=name,
                    version=model_details.version,
                    key=key,
                    value=value
                )
        
        return model_details.version

    
    def register_current_run_model(
        self,
        name: str, 
        description: Optional[str] = None,
        tags: Optional[dict[str, str]] = None
    ) -> str:
        """
        Register a model from the current run to the MLflow Model Registry
        
        Parameters:
        -----------
        name : str
            Name to register the model under
        description : str, optional
            Description for the model version
        tags : Dict[str, str], optional
            Tags to associate with the model version
            
        Returns:
        --------
        str
            The version of the registered model
        """
        if self._run is None:
            raise ValueError("No run is active at the moment.")

        if self._model_uri is None:
            raise ValueError("No model to register.")

        return self.register_model(self._model_uri, name, description, tags)


def register_model(
    model_uri: str,
    name: str,
    description: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
    await_registration_for: Optional[int] = None,
) -> str:
    """
    Register a model from a specified run to the MLflow Model Registry
    
    Parameters:
    -----------
    model_uri: str
        ID of the logged model
    artifact_path : str
        Path to the model artifact within the run
    name : str
        Name to register the model under
    description : str, optional
        Description for the model version
    tags : Dict[str, str], optional
        Tags to associate with the model version
    await_registration_for : int, optional
        Seconds to wait for the model version to become READY
        
    Returns:
    --------
    str
        The version of the registered model
    """
    client = MlflowClient(os.getenv("MLFLOW_TRACKING_URI"))
    
    model_details = mlflow.register_model(
        model_uri=model_uri,
        name=name,
        await_registration_for=await_registration_for
    )
    
    if description is not None:
        client.update_model_version(
            name=name,
            version=model_details.version,
            description=description
        )
    
    if tags is not None:
        for key, value in tags.items():
            client.set_model_version_tag(
                name=name,
                version=model_details.version,
                key=key,
                value=value
            )
    
    return model_details.version
