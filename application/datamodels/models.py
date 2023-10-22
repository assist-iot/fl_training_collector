from enum import Enum
from typing import Optional, Dict, Union, List

from pydantic import BaseModel, Field


class BasicConfiguration(BaseModel):
    config_id: Optional[str]
    batch_size: Optional[int] = 32
    steps_per_epoch: Optional[int] = 3
    epochs: int
    learning_rate: Optional[float] = 0.05


class StatusEnum(str, Enum):
    INACTIVE = 'INACTIVE'
    WAITING = 'WAITING'
    TRAINING = 'TRAINING'
    INTERRUPTED = 'INTERRUPTED'
    FINISHED = 'FINISHED'


class Status(BaseModel):
    status: StatusEnum = StatusEnum.INACTIVE
    round: int = 0
    id: int = -1

    class Config:
        use_enum_values = True


class ServerConfiguration(BaseModel):
    num_rounds: Optional[int]
    round_timeout: Optional[float]


class StrategyConfiguration(BaseModel):
    fraction_fit: Optional[float]
    fraction_evaluate: Optional[float]
    min_fit_clients: Optional[int]  # Minimum number of clients to be sampled for the
    # next round
    min_evaluate_clients: Optional[int]
    min_available_clients: Optional[int]
    accept_failures: Optional[bool]
    server_learning_rate: Optional[float]
    server_momentum: Optional[float]
    min_completion_rate_fit: Optional[float]
    min_completion_rate_evaluate: Optional[float]
    eta: Optional[float]
    eta_l: Optional[float]
    beta_1: Optional[float]
    beta_2: Optional[float]
    tau: Optional[float]
    q_param: Optional[float]
    qffl_learning_rate: Optional[float]

class DPConfiguration(BaseModel):
    num_sampled_clients: int
    init_clip_norm: float = 0.1
    noise_multiplier: float = 1
    server_side_noising: bool = True
    clip_count_stddev:float = None
    clip_norm_target_quantile: float = 0.5
    clip_norm_lr: float = 0.2

class HMConfiguration(BaseModel):
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = [60, 40, 40]
    scale_bits: int = 40
    scheme: str = "CKKS"

class TCTrainingConfiguration(BaseModel):
    strategy: str
    model_name: str
    model_version: str
    adapt_config: Optional[str]
    server_conf: ServerConfiguration
    strategy_conf: StrategyConfiguration
    client_conf: List[BasicConfiguration]
    privacy_mechanisms: Dict[str, Union[DPConfiguration, HMConfiguration]] = Field(..., alias='privacy-mechanisms')
    configuration_id: int
    stopping_flag: Optional[bool] = False
    stopping_target: Optional[Dict[str, float]]

    class Config:
        arbitrary_types_allowed = True
