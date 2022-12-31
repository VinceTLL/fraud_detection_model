import pytest

from fraud_detection_model.config.core import config
from fraud_detection_model.processing.data_manager import load_dataset


@pytest.fixture()
def sample_input_train_data():
    return load_dataset(file_name=config.app_config.training_data_file)


@pytest.fixture()
def sample_input_test_data():
    return load_dataset(file_name=config.app_config.test_data_file)
