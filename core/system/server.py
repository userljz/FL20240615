from functools import reduce
import numpy as np


def aggregate(results):
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


class Server:
    def __init__(self, cfg):
        self.cfg = cfg
    
    @staticmethod
    def aggregate_fit(results):
        if results is None:
            print("In the first round, None results")
            return None
        
        weights_results = [
            (fit_res.parameters, fit_res.num_examples)
            for fit_res in results
        ]
        parameters_aggregated = aggregate(weights_results)
        return parameters_aggregated
    
    def server_conduct(self, results):
        parameters_aggregated = self.aggregate_fit(results)
        return parameters_aggregated
    