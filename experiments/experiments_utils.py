from dataclasses import dataclass
from src import constants

RESULTS_DIR = 'results'


@dataclass
class GenericRunConfig:
    k: int = 10
    dense_size: int = constants.STORM_NETWORK_DEFAULT_DENSE_SIZE
    dropout: float = constants.STORM_NETWORK_DEFAULT_DROPOUT
    lstm_size: int = constants.STORM_NETWORK_DEFAULT_LSTM_SIZE
    use_reg: bool = constants.STORM_NETWORK_DEFAULT_REGULARIZATION_USE


@dataclass
class SingleRunConfig(GenericRunConfig):
    dataset: str = 'blocks'
    use_multirocket: bool = False
    calc_first_order_diff: int = 0
    num_features: int = constants.STORM_DEFAULT_NUM_FEATURES
    block_size: int = constants.STORM_DEFAULT_BLOCK_SIZE

    def __post_init__(self):
        self.method = 'storm multirocket' if self.use_multirocket else 'storm minirocket'
