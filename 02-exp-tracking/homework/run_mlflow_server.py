import click
import os
@click.command()
@click.option('--db-path', default='/workspaces/mlops-zoomcamp/02-exp-tracking/homework/mlflow.db', help='Path to the SQLite database file.')
@click.option('--artifacts-path', default='/workspaces/mlops-zoomcamp/02-exp-tracking/homework/artifacts', help='Path to the folder for artifacts storage.')
@click.option('--host', default='0.0.0.0', help='Host for MLflow server.')
@click.option('--port', default=5000, help='Port for MLflow server.')
def run_mlflow_server(db_path, artifacts_path, host, port):
    command = f"mlflow server " \
              f"--backend-store-uri sqlite:///{db_path} " \
              f"--default-artifact-root {artifacts_path} " \
              f"--host {host} " \
              f"--port {port} &"
    click.echo(f"Running command: {command}")
    os.system(command)
if __name__ == '__main__':
    run_mlflow_server()