import wandb
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from soft_optim.game_generator import generate_dataset


def create_dataset(tokenizer: AutoTokenizer, number_games: int = 10) -> Dataset:
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
    dataset = Dataset.from_dict({"text":list_of_game_strings})
    
    # Tokenize the text prompts (creates "input_ids" property for each dataset item)
    dataset = dataset.map(lambda examples: tokenizer(examples["text"]), batched=True)
    
    # Set the labels to be the same as the input IDs
    dataset = dataset.map(lambda examples: {"labels": examples["input_ids"]}, batched=True)
    
    return dataset


def main(model_name: str = "gpt2") -> None:
    """Fine tune a language model on the games dataset
    
    This is so that our model reliably outputs allowed game moves.
    """
    # Create tokenized datasets (train and eval)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = create_dataset(tokenizer, 5000)
    eval_dataset = create_dataset(tokenizer, 50)
   
    # Initialise Weights & Biases
    wandb.login()

    # Create the model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    training_args = TrainingArguments(
        output_dir=".checkpoints", 
        evaluation_strategy="epoch",
        num_train_epochs=1,
    )
    
    # Fine tune
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    trainer.train()

    # print model output
    out = model.generate(max_length=1000, do_sample=True)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    # Save the final state dictionary

if __name__ == "__main__":
    main()
