import base64
import copy
import os
import pickle
import shutil
from logging import log, INFO
from typing import List, Union

import numpy as np
import requests
import tenseal as ts
from flwr.common import Parameters, NDArrays, parameters_to_ndarrays
from flwr.server.strategy.aggregate import weighted_loss_avg

from application.config import REPOSITORY_ADDRESS, HM_PUBLIC_FILE


def aggregate_metrics(metrics_results):
    """
    A method that does a weighted average of all possible metrics inside
    """
    metrics_inside = [m[1] for m in metrics_results]
    keys = metrics_inside[0].keys()
    results = []
    for length, item in metrics_results:
        single_res = {}
        single_res["length"] = length
        for key in item:
            single_res[key] = item[key]
        results.append(copy.deepcopy(single_res))
    final_res = {key: weighted_loss_avg(
        [
            (result["length"], result[key])
            for result in results
        ]
    ) for key in keys}
    return final_res


def download_strategy(strategy_name: str):
    try:
        with requests.get(f"{REPOSITORY_ADDRESS}/strategy"
                          f"/{strategy_name}",
                          stream=True) as r:
            with open('temp.pkl', 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        with open('temp.pkl', 'rb') as f:
            strategy = pickle.load(f)
        return strategy
    except requests.exceptions.RequestException as e:
        log(INFO, f"Exception while downloading strategy: {e}")
    finally:
        os.remove("temp.pkl")


class HMKeySerializer:

    @staticmethod
    def write(file, data):
        data = base64.b64encode(data)
        with open(file, 'wb') as f:
            f.write(data)

    @staticmethod
    def read(file):
        with open(file, "rb") as f:
            data = f.read()
            return base64.b64decode(data)


class ParamsSerializer:
    '''
    A class for storing the training weights periodically
    '''

    path_end = "aggregated-weights.pkl"

    def store(self, params, path):
        is_exist = os.path.exists(path)
        if not is_exist:
            # Create a new directory because it does not exist
            os.makedirs(path)
        if params.tensor_type == "homomorphic":
            self.store_homomorphic(params, path)
        else:
            self.store_ndarrays(params, path)
        log(INFO, f"Saving final aggregated_weights...")

    def store_ndarrays(self, params, path):
        aggregated_weights = parameters_to_ndarrays(params)
        with open(os.path.join(f"{path}", f"{self.path_end}"), 'wb') as f:
            pickle.dump(aggregated_weights, f)

    def store_homomorphic(self, params, path):
        with open(os.path.join(f"{path}", f"{self.path_end}"), 'wb') as f:
            pickle.dump(params, f)


def ts_tensors_to_parameters(tensors: ts.tensors.ckkstensor.CKKSTensor) -> Parameters:
    """Convert Tenseal tensors to parameters object."""
    tensors = [tensor.serialize() for tensor in tensors]
    return Parameters(tensors=tensors, tensor_type="homomorphic")


def parameters_to_ts_tensors(parameters: Parameters) -> List[ts.tensors.ckkstensor.CKKSTensor]:
    """Convert parameters object to Tenseal tensors."""
    context = ts.context_from(
        HMKeySerializer.read(os.path.join(os.sep, "code", "application", "src", "hm_keys", HM_PUBLIC_FILE)))
    return [ts.ckks_tensor_from(context, tensor) for tensor in parameters.tensors]


def add_gaussian_noise(update: Union[NDArrays, ts.tensors.ckkstensor.CKKSTensor], std_dev: float) -> Union[
        NDArrays, ts.tensors.ckkstensor.CKKSTensor]:
    """Adds iid Gaussian noise of the given standard deviation to each floating
    point value in the update."""
    update_noised = [
        layer + np.random.normal(0, std_dev, layer.shape) for layer in update
    ]
    return update_noised
