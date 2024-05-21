import os
import pickle
import click
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--model_path",
    default="./model",
    help="Location where the trained model will be saved"
)
def run_train(data_path: str, model_path: str):

    # Load the preprocessed training and validation data
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    # Initialize MLflow and enable autologging
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        # Initialize and train the RandomForestRegressor
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)

        # Predict on the validation set
        y_pred = rf.predict(X_val)

        # Calculate and print the RMSE
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(f"Validation RMSE: {rmse}")

        # Create model_path folder unless it already exists
        os.makedirs(model_path, exist_ok=True)

        # Save the trained model
        model_filename = os.path.join(model_path, "random_forest_model.pkl")
        with open(model_filename, "wb") as f_out:
            pickle.dump(rf, f_out)
        print(f"Model saved to {model_filename}")


if __name__ == '__main__':
    run_train()