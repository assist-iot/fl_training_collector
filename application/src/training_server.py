
import logging
import socket

import flwr as fl
from config import FEDERATED_PORT
from datamodels.models import TCTrainingConfiguration
from flwr.server.strategy import Strategy
from src.strategy_manager import TrainingStrategyWrapper
from flwr.common import logger

from application.src.privacy_manager import ServerPrivacyManager
from application.utils import aggregate_metrics, download_strategy


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def construct_strategy(training_id: int, data: TCTrainingConfiguration, num_rounds: int, jobs) \
        -> Strategy:
    strategy = fl.server.strategy.FedAvg() if data.strategy == "avg" else \
        download_strategy(data.strategy)
    strategy.__init__(**data.strategy_conf.dict(exclude_unset=True),
                      fit_metrics_aggregation_fn=aggregate_metrics, evaluate_metrics_aggregation_fn=aggregate_metrics)
    strategy = ServerPrivacyManager().wrap(strategy, data.privacy_mechanisms)
    return TrainingStrategyWrapper(
        strategy=strategy,
        strategy_conf=data.strategy_conf,
        num_rounds=num_rounds,
        training_id=training_id,
        model_name=data.model_name,
        model_version=data.model_version,
        jobs=jobs,
        stopping_flag=data.stopping_flag,
        stopping_target=data.stopping_target,
        configuration_id=data.configuration_id)


def start_flower_server(training_id: int, data: TCTrainingConfiguration, jobs):
    config = fl.server.ServerConfig(
        **data.server_conf.dict(exclude_unset=True))
    strategy = construct_strategy(training_id, data, config.num_rounds, jobs)
    logger.logger.setLevel(logging.INFO)
    fl.server.start_server(config=config,
                           server_address=f"[::]:{FEDERATED_PORT}",
                           strategy=strategy)
