import os
import pickle
import click
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy import sparse
import mlflow
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_train = sparse.csr_matrix(X_train).toarray()  # 转换为密集矩阵 Array
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_val = sparse.csr_matrix(X_val).toarray()  # 转换为密集矩阵 Array
    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"RMSE: {rmse}")
    # 手动记录超参数和指标
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("random_state", 0)
    mlflow.log_metric("rmse", rmse)

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc-green-taxi")
    mlflow.autolog(log_input_examples=False, log_model_signatures=False)
    with mlflow.start_run():
        run_train()