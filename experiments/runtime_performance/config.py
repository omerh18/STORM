from experiments.experiments_utils import GenericRunConfig
from dataclasses import dataclass, field
from typing import List
from src import constants


@dataclass
class RuntimeExperimentRunConfig(GenericRunConfig):
    datasets: str = field(default_factory=lambda: constants.DATASETS_LIST)
    num_epochs: int = 200
    multirocket_usage: List[bool] = field(default_factory=lambda: [False, True])
    first_order_diff_usage: List[int] = field(default_factory=lambda: [0, 1])
    num_features_array: List[int] = field(default_factory=lambda: [1000, 2500, 10000])
    block_size_array: List[int] = field(default_factory=lambda: [1000, 100])
    write_csv: bool = True


run_params_config = RuntimeExperimentRunConfig()
