from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel
from strictyaml import YAML, load

import fraud_detection_model

PACKAGE_ROOT = Path(fraud_detection_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):

    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):

    """
    All configuration relevant to model training and feature engineering.
    """

    target: str
    date_transformation: List[str]
    amt_per_txn_day_partition: List[str]
    avg_day_amt_partition: List[str]
    std_day_amt_partition: List[str]
    fraud_freq_mapper: List[str]
    farm_hash_mapper: List[str]
    log_transformation: List[str]
    varaiable_list: List[str]
    avg_day_amt_date_cols: List[str]
    std_amt_per_txn_year_cols: List[str]
    amt_per_txn_date_cols: List[str]
    seed: int
    class_weight: Dict[int, int]
    aggregation_col: str
    avg_aggregation_type: str
    std_aggregation_type: str
    sum_aggregation_value: str
    features: List[str]
    train_size: float
    random_state: int


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception("Config not found at {}".format(CONFIG_FILE_PATH))


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:

        with open(cfg_path, "r") as config_file:
            parsed_config = load(config_file.read())
            return parsed_config

    raise OSError("Did not find config file at path: {}".format(cfg_path))


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.

    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
