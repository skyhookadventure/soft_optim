import os
import random
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import wandb

from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizer, PreTrainedTokenizerFast,
                          Trainer, TrainingArguments)
from transformers.utils import logging

from soft_optim.game import TicTacToeGame, generate_dataset


def create_dataset(tokenizer: AutoTokenizer,
                   number_games: int = 10) -> Dataset:
    """Create the dataset

    This is a collection of full game prompts (tokenized).

    Args:
        tokenizer: Tokenizer
        number_games: Number of games

    Returns:
        Dataset: Full game prompts dataset
    """
    # Create the dataset from a list of game strings
    list_of_game_strings = generate_dataset(number_games)
    dataset = Dataset.from_dict({"text": list_of_game_strings})

    # Tokenize the text prompts (creates "input_ids" property for each dataset
    # item)
    dataset = dataset.map(
        lambda examples: tokenizer(examples["text"]),  # type: ignore
        batched=True
    )

    # Set the labels to be the same as the input IDs
    dataset = dataset.map(
        lambda examples: {
            "labels": examples["input_ids"]},
        batched=True)

    return dataset


valid_games_fine_tuned_checkpoint = Path(
    __file__).parent / "checkpoints" / "fine_tuned_gpt2"


def fine_tune(
    model: PreTrainedModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    log_weights_and_biases: bool = False,
) -> AutoModelForCausalLM:
    """Fine tune a language model on the games dataset

    This is so that our model reliably outputs allowed game moves.
    """
    # Create tokenized datasets (train and eval)
    train_dataset = create_dataset(tokenizer, 5000)  # type: ignore
    eval_dataset = create_dataset(tokenizer, 50)  # type: ignore

    # Initialise Weights & Biases
    if log_weights_and_biases:
        wandb.login()
        wandb.init(project="soft_optim_fine_tune")

    training_args = TrainingArguments(
        output_dir=".checkpoints",
        evaluation_strategy="epoch",
        num_train_epochs=1,
        seed=0,
        data_seed=0
    )

    # Fine tune
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore

    )
    trainer.train()

    # print model output
    out = model.generate(max_length=1000, do_sample=True)  # type: ignore
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    return model


def infer_game(
    model: AutoModelForCausalLM,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    num_samples: int = 1
) -> List[str]:
    """Infer a full game from just the start text

    Args:
        model: Pretrained model
        tokenizer: Tokenizer
        num_samples: Number of samples to generate

    Returns:
        bool: All games are valid
    """
    n = num_samples
    game_start_text = "Let's play Tic Tac Toe:\n"
    tokens = tokenizer(
        [game_start_text] *
        n,
        return_tensors="pt").to(
        model.device)
    out = model.generate(**tokens, max_length=1000, do_sample=True)
    samples = tokenizer.batch_decode(out, skip_special_tokens=True)

    # Get just the board states
    stripped_samples = []

    for full_game in samples:
        game = TicTacToeGame()
        stripped_samples.append(game.extract_game_string(full_game))

    return stripped_samples


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Set logging
    logging.set_verbosity_error()
    # logging.disable_progress_bar()
    wandb.init(mode="disabled")
    os.environ["WANDB_DISABLED"] = "true"

    # Create the model
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    isValid: bool = False

    # Train until
    while not isValid:
        fine_tune(model, tokenizer, False)

        games = infer_game(model, tokenizer, 20)
        valid_games: List[bool] = []

        for full_game in games:
            game = TicTacToeGame()
            game_is_valid, a, b = game.validate_game_string(full_game)
            valid_games.append(game_is_valid)

            if not game_is_valid:
                print(full_game)

        print(f"Is valid:", str(valid_games))

        isValid = all(valid_games)

    # Save the model
    if not valid_games_fine_tuned_checkpoint.parent.exists():
        os.mkdir(valid_games_fine_tuned_checkpoint)

    model.save_pretrained(valid_games_fine_tuned_checkpoint)  # type: ignore
