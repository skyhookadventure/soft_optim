import numpy as np


class TestEmpiricalErrorBound:
    def returns_correct_bound(self):
        number_samples = 100
        mock_proxy_reward = np.random(number_samples)
        mock_human_evaluated_reward = np.random(number_samples)