# mlflow_test.py
import mlflow

print("CWD:", __import__("os").getcwd())

mlflow.set_experiment("test_experiment")

with mlflow.start_run(run_name="test_run"):
    mlflow.log_param("param1", 123)
    mlflow.log_metric("metric1", 0.456)

print("Done logging.")