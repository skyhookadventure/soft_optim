from typing import Dict, List, Optional

from game import TicTacToeGame


def metrics(
    samples: List[str],
    prompts: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None
) -> Dict[str, List[float]]:
    """Metrics

    Args:
        samples: Batch of responses
        prompts: Batch of prompts
        outputs: Batch of outputs

    Returns:
        Dict[str, List[float]]: Dict of metrics, where the key is the metric
        name and the value is a list of metric values (one for each item in the
        batch).
    """
    true_rewards: List[float] = []
    valid_games: List[float] = []

    for s in samples:
        g = TicTacToeGame(check_valid_move=True, check_valid_state=True)
        true_rewards.append(g.evaluate_game_string(s))
        isValid: bool = g.validate_game_string(s)[0]
        valid_games.append(1.0 if isValid else 0.0)

    return {"true_reward": true_rewards, "is_valid": valid_games}
