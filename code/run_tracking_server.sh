mlflow server \
  --backend-store-uri sqlite:///final_experiments.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 --port 8080
