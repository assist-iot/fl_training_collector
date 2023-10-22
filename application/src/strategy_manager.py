
import os
import pickle
from logging import INFO
import shutil
import sys
import traceback
from typing import List, Tuple, Optional, Dict, Union

import flwr as fl
import requests
from config import JSON_FILE
from datamodels.models import Status, StatusEnum
from flwr.common import Scalar, Parameters, EvaluateRes, MetricsAggregationFn, FitIns, EvaluateIns
from flwr.common.logger import log
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from application.config import ORCHESTRATOR_ADDRESS, REPOSITORY_ADDRESS
from application.datamodels.models import StrategyConfiguration
from application.utils import ParamsSerializer


class TrainingStrategyWrapper(fl.server.strategy.Strategy):
    """Wrapper for configuring a strategy with additional behaviours"""

    def __init__(
            self,
            strategy: fl.server.strategy.Strategy,
            strategy_conf: StrategyConfiguration,
            num_rounds: int,
            model_name: str,
            model_version: str,
            jobs,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            training_id: int = -1,
            configuration_id: int = -1,
            stopping_flag: bool = False,
            stopping_target: Dict[str, float] = {}
    ) -> None:
        self.strategy = strategy
        self.strategy_conf = strategy_conf
        self.num_rounds = num_rounds
        self.training_id = training_id
        self.configuration_id = configuration_id
        self.model_name = model_name
        self.model_version = model_version
        self.path = os.path.join("model", f"{self.training_id}")
        self.serializer = ParamsSerializer()
        self.stopping_flag = stopping_flag
        self.stopping_target = stopping_target
        self.jobs = jobs
        self.jobs[self.training_id].status = StatusEnum.WAITING
        self.jobs[self.training_id].round = 0

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        return self.strategy.configure_fit(
            server_round, parameters, client_manager
        )

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def send_training_results(self, loss, additional, server_round):
        with open(os.path.join(f"{self.path}", "aggregated-weights.pkl"), 'rb') as f:
            try:
                to_send = {"model_name": f"{self.model_name}",
                           "model_version": f"{self.model_version}",
                           "training_id": f"{self.training_id}",
                           "results": {"rounds": f"{server_round}",
                                       "final_loss": f"{loss}",
                                       **additional,
                                       **self.strategy_conf.dict(exclude_unset=True)},
                           "configuration_id": f"{self.configuration_id}"}
                log(INFO, f"Final results to send look like {to_send}")
                r = requests.post(f"{REPOSITORY_ADDRESS}/training-results",
                                  json=to_send)
                r = requests.put(f"{REPOSITORY_ADDRESS}/training-results"
                                 f"/{self.model_name}/{self.model_version}"
                                 f"/{self.training_id}/{self.configuration_id}",
                                 files={"file": f})
            except requests.exceptions.RequestException as e:
                print(f"Failed to send the training results of job {self.training_id} to "
                      f"repository")
                traceback.print_exc()
        shutil.rmtree(f"{self.path}")

    def send_early_end_notice(self, metric, additional, server_round):
        try:
            log(INFO,
                f"The TC has achieved the metric of {metric} of {additional[metric]} which is larger than {self.stopping_target[metric]}")
            r = requests.post(f"{ORCHESTRATOR_ADDRESS}/FL_early_end",
                              json={"model_name": self.model_name, "model_version":
                                    self.model_version, "round": server_round, "metric": metric, "value_achieved": additional[metric],
                                    "value_maximal": self.stopping_target[metric]})
        except requests.exceptions.ConnectionError as e:
            print(f"Failed to inform the Orchestrator about early end")

    def finish_training(self, loss, additional, server_round):
        self.jobs[self.training_id].status = StatusEnum.FINISHED
        self.jobs[self.training_id].round = server_round
        self.send_training_results(loss, additional, server_round)

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_params, metrics = self.strategy.aggregate_fit(server_round, results,
                                                                 failures)
        self.jobs[self.training_id].status = StatusEnum.TRAINING
        self.jobs[self.training_id].round = server_round
        if aggregated_params is not None:
            self.serializer.store(params=aggregated_params, path=self.path)
        return aggregated_params, metrics

    def aggregate_evaluate(self,
                           server_round: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
                           ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        loss, additional = self.strategy.aggregate_evaluate(server_round, results,
                                                            failures)
        log(INFO,
            f"Aggregated metrics for round {server_round} are {additional}")
        if self.stopping_flag and self.stopping_target:
            for to_check in self.stopping_target:
                # Check whether the metric was surpassed in a given evaluation round
                if to_check in additional and additional[to_check] > self.stopping_target[to_check]:
                    self.finish_training(loss, additional, server_round)
                    self.send_early_end_notice(
                        to_check, additional, server_round)
                    sys.exit()
        if server_round == self.num_rounds:
            self.finish_training(loss, additional, server_round)
        with open(os.path.join("..", JSON_FILE), 'wb') as handle:
            pickle.dump(self.jobs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            r = requests.post(f"{ORCHESTRATOR_ADDRESS}/FL_traininground",
                              json={"model_name": self.model_name, "model_version":
                                    self.model_version, "round": server_round})
        except requests.exceptions.ConnectionError as e:
            log(INFO,
                f'Could not connect to orchestrator on {ORCHESTRATOR_ADDRESS}')
        return loss, additional

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return self.strategy.evaluate(server_round, parameters)
