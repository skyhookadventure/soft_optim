import numpy as np
from typing import List

from soft_optim.quantilizer import empirical_error_bound

class TestEmpiricalErrorBound:
    def test_error_bounds_work_with_lots_samples(self):
        # Set the random seed to avoid a brittle test
        np.random.seed(0)
        
        epsilon: float = 0.05
        number_distributions_compare: int = 1000
        experienced_outside_of_bounds: List[bool] = []
        
        # For x meta-samples:
        for _ in range(number_distributions_compare):
            # Highest variance we can have occurs with a mean of 0.5 & bernoulli
            # distribution 
            error_distribution_mean: float = np.random.uniform(0, 1)
            sample_errors = np.random.binomial(1, error_distribution_mean, 1000)
            
            error_bound: float = empirical_error_bound(
                np.zeros(1000),
                sample_errors,
                epsilon)
            
            # See if we were in the error bound
            is_outside_bounds: bool = error_bound < error_distribution_mean
            experienced_outside_of_bounds.append(is_outside_bounds)
           
        # Check epsilon percent of x samples are outside the error bound
        assert np.mean(experienced_outside_of_bounds) < epsilon
            