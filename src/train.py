import os
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

MODEL_NAME = os.environ.get("BASE_MODEL", "distilgpt2")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/distilgpt2-dialogue")


def load_and_prepare_dataset():
    # Using DailyDialog from datasets
    ds = load_dataset("daily_dialog")
    # concatenate conversations into text lines
    def join_turns(example):
        # `dialog` is a list of utterances
        example["text"] = "\n".join(example["dialog"]) + "\n"
        return example
    ds = ds.map(join_turns)
    return ds["train"].train_test_split(test_size=0.05)


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=512)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    data = load_and_prepare_dataset()
    tokenized = data.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["dialog", "act", "emotion", "text"])

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        logging_steps=100,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        fp16=False,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()
