from pathlib import Path

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from soft_optim.fine_tune import (create_dataset, fine_tune, infer_game,
                                  valid_games_fine_tuned_checkpoint)
from soft_optim.game import TicTacToeGame


class TestCreateDataset:
    def test_tokenizes_text(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        dataset = create_dataset(tokenizer, 1)
        first_example = next(iter(dataset))
        text = first_example["text"]
        input_ids = first_example["input_ids"]
        expected_input_ids = tokenizer.encode(text)
        assert input_ids == expected_input_ids

    def test_adds_labels(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        dataset = create_dataset(tokenizer, 1)
        first_example = next(iter(dataset))
        input_ids = first_example["input_ids"]
        labels = first_example["labels"]
        assert input_ids == labels


class TestCheckModelOutputsValidGame:
    def test_plain_gpt(self):
        # Load standard GPT2 (not fine tuned)
        model_name = "gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Infer the full game
        full_game: str = infer_game(model, tokenizer)

        # Check it throws an error
        with pytest.raises(Exception) as exc_info:
            TicTacToeGame.evaluate_game_string(None, full_game)

        assert exc_info

    def test_fine_tuned_gpt(self):
        # Run the model if it hasn't already been run
        if not valid_games_fine_tuned_checkpoint.exists():
            fine_tune(log_weights_and_biases=False)

        # Load the fine-tuned model
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            valid_games_fine_tuned_checkpoint)

        # Infer the game
        full_game: str = infer_game(model, tokenizer)

        # Check it is valid
        res = evaluate_game_string(full_game)
        assert type(res) == int
