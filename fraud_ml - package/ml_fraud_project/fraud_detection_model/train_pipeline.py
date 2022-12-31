from config.core import config
from pipeline import fraud_pipe
from processing.data_manager import load_dataset, save_pipeline


def run_training() -> None:
    """Train the model."""

    # read training data

    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test

    # end = round(len(data) * config.model_config.train_size)
    # X_train = data[config.model_config.features][:end]
    # y_train = data[config.model_config.target][:end]
    # X_test = data[config.model_config.features][end:]
    # y_test = data[config.model_config.target][end:]

    # fit model
    X_train = data[config.model_config.features]
    y_train = data[config.model_config.target]

    fraud_pipe.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=fraud_pipe)


if __name__ == "__main__":
    run_training()
