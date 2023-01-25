import trlx
from trlx.data.configs import TRLConfig
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from game import TicTacToeGame
import soft_optim.quantilizer as quantilizer
import numpy as np
from typing import List, Dict, Optional
import wandb
import traceback

from soft_optim.fine_tune import valid_games_fine_tuned_checkpoint, infer_game


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


def no_soft_opt_experiment():
    def reward_fn(samples, prompts=None, outputs=None):
        rewards = []
        g = TicTacToeGame(check_valid_move=False, check_valid_state=False)
        for s in samples:
            rewards.append(g.evaluate_game_string(s))
        return rewards

    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    config_path = Path(__file__).parent / "configs/ppo_gpt2.yml"
    config = TRLConfig.load_yaml(config_path)

    # collect a tictactoe generator model that was trained with fine_tune.py
    model_path = valid_games_fine_tuned_checkpoint

    trainer = trlx.train(
        str(model_path),
        reward_fn=reward_fn,
        config=config,
        prompts=["Let's play Tic Tac Toe:"] * config.train.batch_size,
        metric_fn=metrics,

    )

    # test model output
    game_start_text = "Let's play Tic Tac Toe:\n"
    tokens = tokenizer.encode(game_start_text, return_tensors="pt").to('cuda')
    out = trainer.model.generate(tokens, max_length=1000, do_sample=True)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    fine_tuned_model_path = Path(__file__).parent / \
        ".checkpoints" / "no_soft_opt_model"
    trainer.save(fine_tuned_model_path)


def soft_opt_experiment(kl_setting=1.0, lr=1e-5, epochs=100):
    wandb.login()
    wandb.init(
        project="soft_optim",
        name=f"mod-kl v1 soft_optim_experiment kl={kl_setting}, lr={lr}, epochs={epochs}",
        reinit=True,)
    # wandb.config.update(allow_val_change=True)

    # collect a tictactoe generator model that was trained with fine_tune.py
    model_path = valid_games_fine_tuned_checkpoint
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # load model
    model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')

    # get samples for gen error calculation
    samples = infer_game(model, tokenizer, num_samples=200)
    proxy_rewards: List[float] = []
    human_rewards: List[float] = []
    g_proxy = TicTacToeGame(check_valid_move=False, check_valid_state=False)
    g_human = TicTacToeGame()
    for s in samples:
        proxy_rewards.append(g_proxy.evaluate_game_string(s))
        human_rewards.append(g_human.evaluate_game_string(s))
    proxy_rewards_arr = np.array(proxy_rewards)
    human_rewards_arr = np.array(human_rewards)

    # get generalization error
    eps = 0.05  # <5% chance of bound being exceeded
    bound = quantilizer.empirical_error_bound(
        proxy_rewards_arr, human_rewards_arr, eps)
    # work out proxy reward cutoff
    cutoff = quantilizer.get_proxy_value_cutoff(
        bound, len(samples), model, tokenizer)

    print(bound)
    print(cutoff)

    def loglikelihood_approx(rewards, cutoff):
        alpha = 30.0  # hyperparameter determining sharpness of cutoff
        return np.log(1 / (1 + np.exp(-alpha * (rewards - cutoff))))

    # def loglikelihood_approx(rewards, cutoff):
    #    return np.log10((rewards > cutoff)+1e-8)

    def reward_fn(samples, prompts=None, outputs=None):
        rewards = []
        g = TicTacToeGame(check_valid_move=False, check_valid_state=False)
        for s in samples:
            rewards.append(g.evaluate_game_string(s))
        rewards_arr = np.array(rewards)
        return loglikelihood_approx(rewards_arr, cutoff)

    config_path = Path(__file__).parent / "configs/ppo_gpt2.yml"
    config = TRLConfig.load_yaml(config_path)

    # custom config options for this experiment
    config.method.target = None  # Set to constant KL penalty
    config.method.init_kl_coef = kl_setting  # 1.0  # set weight of KL penalty to 1
    config.train.epochs = epochs
    config.optimizer.kwargs["lr"] = lr

    trainer = trlx.train(
        str(model_path),
        reward_fn=reward_fn,
        config=config,
        prompts=["Let's play Tic Tac Toe:"] * config.train.batch_size,
        metric_fn=metrics,
    )

    # test model output
    game_start_text = "Let's play Tic Tac Toe:\n"
    tokens = tokenizer.encode(game_start_text, return_tensors="pt").to('cuda')
    out = trainer.model.generate(tokens, max_length=1000, do_sample=True)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    fine_tuned_model_path = Path(__file__).parent / \
        ".checkpoints" / "soft_opt_model"
    trainer.save(fine_tuned_model_path)
    wandb.finish()


if __name__ == "__main__":
    # no_soft_opt_experiment()
    for kl in [0.001, 0.01, 0.1, 0.5, 1, 10]:
        for lr in [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
            for epochs in [400]:
                soft_opt_experiment(kl_setting=kl, lr=lr, epochs=epochs)
    '''
    for kl, lr, epochs in [
            (0.3, 1e-6, 100),
            (0.4, 1e-5, 100),
            (0.4, 1e-5, 1000),
            (0.4, 1e-4, 100),
            (0.7, 1e-5, 100),
            (0.7, 1e-5, 1000),
            (0.7, 1e-6, 1000),
            (1.0, 1e-6, 1000),
            (1.0, 1e-7, 1000),
            (0.1, 1e-5, 1000),
            (0.1, 1e-6, 1000),
            (0.1, 1e-7, 1000),
    ]:
        # try:
        soft_opt_experiment(kl_setting=kl, lr=lr, epochs=epochs)
        # except BaseException as e:
        #    print(e)
        #    print(traceback.format_exc())
    '''
