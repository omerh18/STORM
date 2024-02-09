from experiments.experiments_utils import GenericRunConfig
from dataclasses import dataclass, field
from typing import List
from src import constants


@dataclass
class ClassificationExperimentRunConfig(GenericRunConfig):
    datasets: str = field(default_factory=lambda: constants.DATASETS_LIST)
    num_epochs: int = 200
    multirocket_usage: List[bool] = field(default_factory=lambda: [False, True])
    first_order_diff_usage: List[int] = field(default_factory=lambda: [0, 1])
    num_features: int = constants.STORM_DEFAULT_NUM_FEATURES
    block_size: int = constants.STORM_DEFAULT_BLOCK_SIZE
    write_csv: bool = True


num_features_block_size_combinations = [
    {
        'num_features': 1000,
        'block_size': 1000
    },
    {
        'num_features': 1000,
        'block_size': 100
    },
    {
        'num_features': 2500,
        'block_size': 10000
    },
    {
        'num_features': 2500,
        'block_size': 1000
    },
    {
        'num_features': 2500,
        'block_size': 100
    },
    {
        'num_features': 10000,
        'block_size': 1000
    },
    {
        'num_features': 10000,
        'block_size': 100
    }
]

run_params_configs = []
for combination in num_features_block_size_combinations:
    run_params_configs.append(
        ClassificationExperimentRunConfig(
            num_features=combination['num_features'],
            block_size=combination['block_size']
        )
    )
