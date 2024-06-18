from functools import reduce
import numpy as np


# def aggregate(results):
#     """Compute weighted average."""
#     # Calculate the total number of examples used during training
#     num_examples_total = sum([num_examples for _, num_examples in results])

#     # Create a list of weights, each multiplied by the related number of examples
#     weighted_weights = [
#         [layer * num_examples for layer in weights] for weights, num_examples in results
#     ]

#     # Compute average weights of each layer
#     weights_prime = [
#         reduce(np.add, layer_updates) / num_examples_total
#         for layer_updates in zip(*weighted_weights)
#     ]
#     return weights_prime

def aggregate(results):
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Extract the parameter names from the first set of parameters
    param_names = [name for name, _ in results[0][0]]

    # Initialize a dictionary to accumulate the weighted sums for each parameter
    weighted_sums = {name: 0 for name in param_names}

    # Accumulate the weighted sums
    for weights, num_examples in results:
        for name, value in weights:
            weighted_sums[name] += value * num_examples

    # Compute the average for each parameter
    averaged_weights = {
        name: weighted_sums[name] / num_examples_total for name in param_names
    }

    # Convert the dictionary to a list of (name, value) tuples
    weights_prime = [(name, averaged_weights[name]) for name in param_names]

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
    