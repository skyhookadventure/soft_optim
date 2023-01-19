from typing import List, Callable
import numpy as np


def empirical_error_bound(
    proxy_reward: np.array,  
    human_evaluated_reward: np.array,
    epsilon: float) -> float:
    """Empirical error bound

    Uses Hoeffding's inequality (one-sided bound)
    https://en.wikipedia.org/wiki/Hoeffding%27s_inequality

    Args:
        proxy_reward: The reward function
        human_evaluated_reward: Human evaluation of each game, as e.g 1
        if it didn't break the rules and won, 0 if it did break the rules or
        didn't win.
        epsilon: The probability of the error bound being exceeded.

    Returns:
        float: Error bound (expected difference between human and proxy reward
        evaluations). 
    """
    # Expected difference between the two rewards
    expected_difference: float = np.abs(proxy_reward - human_evaluated_reward).mean()
    
    number_samples = len(proxy_reward)
    
    return expected_difference + np.sqrt(-np.log(epsilon) / (2 * number_samples))
    

    
    