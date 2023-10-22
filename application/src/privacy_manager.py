import logging
import math
from logging import WARNING, INFO
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar, NDArrays, MetricsAggregationFn
from flwr.common.dp import add_gaussian_noise
from flwr.common import logger
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.server.strategy.fedavg import WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW
from flwr.server.strategy.strategy import Strategy

from application.datamodels.models import DPConfiguration, HMConfiguration
from application.utils import parameters_to_ts_tensors, ts_tensors_to_parameters


class ServerPrivacyManager:

    def wrap(self, strategy: Strategy, priv_configuration: Dict[str, Union[HMConfiguration, DPConfiguration]]):
        logger.logger.setLevel(logging.INFO)
        if "homomorphic" in priv_configuration:
            logger.logger.log(logging.INFO, f'Homomorphic encryption set')
            strategy = self.homomorphic_wrap(strategy, priv_configuration["homomorphic"])
        if "dp-adaptive" in priv_configuration:
            logger.logger.log(logging.INFO, f'Differential privacy set')
            strategy = self.dp_wrap(strategy, priv_configuration["dp-adaptive"])
        return strategy

    @staticmethod
    def dp_wrap(strategy: Strategy, config: DPConfiguration):
        input_conf = config.dict()
        return DPFedAvgAdaptive(strategy=strategy, **input_conf)

    @staticmethod
    def homomorphic_wrap(strategy: Strategy, config: HMConfiguration):
        strategy_config = dict(strategy.__dict__)
        strategy_config.pop('self', None)
        hm_fed_keys = HMFedAvg.__init__.__code__.co_varnames
        input = {key: val for key, val in strategy_config.items() if key in hm_fed_keys}
        new_strategy = HMFedAvg(
            **input
        )
        return new_strategy


class DPFedAvgFixed(Strategy):
    """Wrapper for configuring a Strategy for DP with Fixed Clipping."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
            self,
            strategy: Strategy,
            num_sampled_clients: int,
            clip_norm: float,
            noise_multiplier: float = 1,
            server_side_noising: bool = True,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        # Doing fixed-size subsampling as in https://arxiv.org/abs/1905.03871.
        self.num_sampled_clients = num_sampled_clients

        if clip_norm <= 0:
            raise Exception("The clipping threshold should be a positive value.")
        self.clip_norm = clip_norm

        if noise_multiplier < 0:
            raise Exception("The noise multiplier should be a non-negative value.")
        self.noise_multiplier = noise_multiplier

        self.server_side_noising = server_side_noising

    def __repr__(self) -> str:
        rep = "Strategy with DP with Fixed Clipping enabled."
        return rep

    def _calc_client_noise_stddev(self) -> float:
        self.noise_multiplier = np.real_if_close(self.noise_multiplier)
        return float(
            self.noise_multiplier * self.clip_norm / (self.num_sampled_clients ** (0.5))
        )

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        additional_config = {"dpfedavg_clip_norm": self.clip_norm}
        if not self.server_side_noising:
            additional_config[
                "dpfedavg_noise_stddev"
            ] = self._calc_client_noise_stddev()

        client_instructions = self.strategy.configure_fit(
            server_round, parameters, client_manager
        )

        for _, fit_ins in client_instructions:
            fit_ins.config.update(additional_config)

        return client_instructions

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            return None, {}
        # Forcing unweighted aggregation, as in https://arxiv.org/abs/1905.03871.
        for _, fit_res in results:
            fit_res.num_examples = 1
            if isinstance(self.strategy, HMFedAvg):
                noise_coeff = self._calc_client_noise_stddev()
                noised_params = add_gaussian_noise(parameters_to_ts_tensors(fit_res.parameters), noise_coeff)
                fit_res.parameters = ts_tensors_to_parameters(noised_params)
            else:
                fit_res.parameters = ndarrays_to_parameters(
                    add_gaussian_noise(
                        parameters_to_ndarrays(fit_res.parameters),
                        self._calc_client_noise_stddev(),
                    )
                )

        return self.strategy.aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return self.strategy.evaluate(server_round, parameters)


class DPFedAvgAdaptive(DPFedAvgFixed):
    """Wrapper for configuring a Strategy for DP with Adaptive Clipping."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
            self,
            strategy: Strategy,
            num_sampled_clients: int,
            init_clip_norm: float = 0.1,
            noise_multiplier: float = 1,
            server_side_noising: bool = True,
            clip_norm_lr: float = 0.2,
            clip_norm_target_quantile: float = 0.5,
            clip_count_stddev: Optional[float] = None,
    ) -> None:
        super().__init__(
            strategy=strategy,
            num_sampled_clients=num_sampled_clients,
            clip_norm=init_clip_norm,
            noise_multiplier=noise_multiplier,
            server_side_noising=server_side_noising,
        )
        self.clip_norm_lr = clip_norm_lr
        self.clip_norm_target_quantile = clip_norm_target_quantile
        self.clip_count_stddev = clip_count_stddev
        if self.clip_count_stddev is None:
            self.clip_count_stddev = 0
            if noise_multiplier > 0:
                self.clip_count_stddev = self.num_sampled_clients / 20.0

        if noise_multiplier:
            self.noise_multiplier = (
                                            self.noise_multiplier ** (-2) - (2 * self.clip_count_stddev) ** (-2)
                                    ) ** (-0.5)

    def __repr__(self) -> str:
        rep = "Strategy with DP with Adaptive Clipping enabled."
        return rep

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        additional_config = {"dpfedavg_adaptive_clip_enabled": True}

        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )

        for _, fit_ins in client_instructions:
            fit_ins.config.update(additional_config)

        return client_instructions

    def _update_clip_norm(self, results: List[Tuple[ClientProxy, FitRes]]) -> None:
        # Calculating number of clients which set the norm indicator bit
        norm_bit_set_count = 0
        for client_proxy, fit_res in results:
            if "dpfedavg_norm_bit" not in fit_res.metrics:
                raise Exception(
                    f"Indicator bit not returned by client with id {client_proxy.cid}."
                )
            if fit_res.metrics["dpfedavg_norm_bit"]:
                norm_bit_set_count += 1
        # Noising the count
        noised_norm_bit_set_count = float(
            np.random.normal(norm_bit_set_count, self.clip_count_stddev)
        )

        noised_norm_bit_set_fraction = noised_norm_bit_set_count / len(results)
        # Geometric update
        self.clip_norm *= math.exp(
            -self.clip_norm_lr
            * (noised_norm_bit_set_fraction - self.clip_norm_target_quantile)
        )

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            return None, {}
        new_global_model = super().aggregate_fit(server_round, results, failures)
        self._update_clip_norm(results)
        return new_global_model


class HMFedAvg(Strategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        """Federated Averaging strategy modified to be able to work for homomorphic encryption
        """
        super().__init__()

        if (
                min_fit_clients > min_available_clients
                or min_evaluate_clients > min_available_clients
        ):
            logger.log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)
        # log all messages, debug and up
        logger.logger.setLevel(INFO)
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_tensors = parameters_to_ts_tensors(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_tensors, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        weight_results = [
            (parameters_to_ts_tensors(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        norm_base = sum([i for _, i in weight_results])
        weights_multiplied = []
        for weights, num_examples in weight_results:
            multiplier = num_examples/norm_base
            new = [l * multiplier for l in weights]
            weights_multiplied.append(new)
        added = weights_multiplied[0]
        if len(weights_multiplied) > 1:
            for l in weights_multiplied[1:]:
                added = [added[i]+l[i] for i in range(len(added))]
        parameters_aggregated = ts_tensors_to_parameters(added)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            logger.log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            logger.log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
