# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 13:52:19 2025

@author: dhars
""" 
import shutil
import torch
import numpy as np
from torch.utils.data import DataLoader
from nltk.metrics.distance import edit_distance
import matplotlib.pyplot as plt

import lightning as L
import re

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

from transformers import AutoProcessor
from transformers import PaliGemmaForConditionalGeneration

from torch.utils.data import Dataset
from typing import Any, List, Dict
import random
import json
from datasets import load_dataset

from huggingface_hub import login
login("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

import wandb
wandb.init(mode="offline")



# Configuration Constants
MAX_LENGTH = 512
REPO_ID = "google/paligemma-3b-mix-224"
#FINETUNED_MODEL_ID = "xxxxx"
PROMPT = "extract JSON."
WANDB_PROJECT = "paligemma_finetuning"
WANDB_NAME = "paligemma_train"


class CustomDataset(Dataset):

    def __init__(
        self,
        dataset_name_or_path: str,
        split: str = "train",
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.split = split
        self.sort_json_key = sort_json_key

        self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        for sample in self.dataset:
            ground_truth = json.loads(sample["ground_truth"])
            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            self.gt_token_sequences.append(
                [
                    self.json2token(
                        gt_json,
                        sort_json_key=self.sort_json_key,
                    )
                    for gt_json in gt_jsons  # load json from list of json
                ]
            )

    def json2token(self, obj: Any, sort_json_key: bool = True):
        # Convert an ordered JSON object into a token sequence..
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            return obj

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
    
        sample = self.dataset[idx]

        # inputs
        image = sample["image"]
        target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        print(target_sequence)
        return image, target_sequence

class PaliGemmaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

        # Initialize lists to track losses for plotting
        self.train_losses = []
        self.val_losses = []

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, pixel_values, labels = batch

        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             pixel_values=pixel_values,
                             labels=labels)
        loss = outputs.loss

        self.log("train_loss", loss)

        # Append to train_losses list for later plotting
        self.train_losses.append(loss.item())

        # Log to file
        with open("training_loss.txt", "a") as f:
            f.write(f"Train Loss (Epoch {self.current_epoch}): {loss.item()}\n")

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        input_ids, attention_mask, pixel_values, answers = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            pixel_values=pixel_values, max_new_tokens=MAX_LENGTH)
        # turn them back into text, chopping off the prompt
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        val_loss = np.mean(scores)
        self.log("val_edit_distance", val_loss)

        # Append to val_losses list for later plotting
        self.val_losses.append(val_loss)

        # Log to file
        with open("validation_loss.txt", "a") as f:
            f.write(f"Validation Loss (Epoch {self.current_epoch}): {val_loss}\n")

        return scores

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))
        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def on_epoch_end(self):
        # Plot the losses at the end of each epoch
        if self.current_epoch % 1 == 0:  # Change the modulo if you want to plot after specific epochs
            self.plot_losses()

    def plot_losses(self):
        # Plot the training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"loss_plot_epoch_{self.current_epoch}.png")  # Save plot as image
        plt.show() 
        

# Initialize Processor and Model
def initialize_processor_and_model():
    processor = AutoProcessor.from_pretrained(REPO_ID)
    model = PaliGemmaForConditionalGeneration.from_pretrained(REPO_ID)
    return processor, model

# Data Loading
def load_data():
    dataset = load_dataset("naver-clova-ix/cord-v2")
    train_dataset = CustomDataset("naver-clova-ix/cord-v2", split="train")
    val_dataset = CustomDataset("naver-clova-ix/cord-v2", split="validation")
    return dataset, train_dataset, val_dataset

# Save Model & Processor
def save_model(model_module, save_directory):
    model_module.model.save_pretrained(save_directory)
    processor.save_pretrained(save_directory)
    zip_path = f"{save_directory}.zip"
    shutil.make_archive(zip_path.replace(".zip", ""), 'zip', save_directory)
    return zip_path

def train_collate_fn(examples):
  images = [example[0] for example in examples]
  print(images)
  texts = [PROMPT for _ in range(len(images))]
  print(texts)
  labels = [example[1] for example in examples]
  print(labels)

  inputs = processor(text=texts, images=images, suffix=labels, return_tensors="pt", padding=True,
                     truncation="only_second", max_length=MAX_LENGTH,
                     tokenize_newline_separately=False)

  input_ids = inputs["input_ids"]
  token_type_ids = inputs["token_type_ids"]
  attention_mask = inputs["attention_mask"]
  pixel_values = inputs["pixel_values"]
  labels = inputs["labels"]

  return input_ids, token_type_ids, attention_mask, pixel_values, labels


def eval_collate_fn(examples):
  images = [example[0] for example in examples]
  texts = [PROMPT for _ in range(len(images))]
  answers = [example[1] for example in examples]

  inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, tokenize_newline_separately=False)

  input_ids = inputs["input_ids"]
  attention_mask = inputs["attention_mask"]
  pixel_values = inputs["pixel_values"]

  return input_ids, attention_mask, pixel_values, answers


def token2json(tokens, is_inner_value=False, added_vocab=None):
        """
        Convert a (generated) token sequence into an ordered JSON format.
        """
        if added_vocab is None:
            added_vocab = processor.tokenizer.get_added_vocab()

        output = {}

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            key_escaped = re.escape(key)

            end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(
                    f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE | re.DOTALL
                )
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = token2json(content, is_inner_value=True, added_vocab=added_vocab)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + token2json(tokens[6:], is_inner_value=True, added_vocab=added_vocab)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}


def infer(test_example, model, processor):
    test_image = test_example["image"]
    inputs = processor(text=PROMPT, images=test_image, return_tensors="pt")
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)

    # Next turn each predicted token ID back into a string using the decode method
    # chop of the prompt, which consists of image tokens and our text prompt
    image_token_index = model.config.image_token_index
    num_image_tokens = len(generated_ids[generated_ids==image_token_index])
    num_text_tokens = len(processor.tokenizer.encode(PROMPT))
    num_prompt_tokens = num_image_tokens + num_text_tokens + 2
    generated_text = processor.batch_decode(generated_ids[:, num_prompt_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print('generated_text', generated_text)
    return generated_text[0] 


def configure_lora():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16
    )

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    return bnb_config, lora_config 

if __name__ == "__main__":
    processor, model = initialize_processor_and_model()
    bnb_config, lora_config = configure_lora()
    
    # Load the model quantized model..
    model = PaliGemmaForConditionalGeneration.from_pretrained(REPO_ID, quantization_config=bnb_config, device_map={"": 0})
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    config = {"max_epochs": 1,
          # "val_check_interval": 0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 8,
          "lr": 1e-4,
          "batch_size": 2,
          # "seed":2022,
          "num_nodes": 1,
          "warmup_steps": 50,
          "result_path": "./result",
          "verbose": True,} 
    
    
    # Load datasets
    dataset, train_dataset, val_dataset = load_data()
    model_module = PaliGemmaModelPLModule(config, processor, model)
    
    # define model checkpoint..
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",  # Path to save checkpoints
        filename="best_model-{epoch}-{val_edit_distance:.4f}",  # Checkpoint filename format
        monitor="val_edit_distance",  # Metric to monitor for saving the best model
        save_top_k=3,  # Save the top 3 models based on validation metric
        mode="min",  # Lower validation loss is better (for "val_edit_distance")
        save_last=True,  # Save the last model checkpoint
    )
    
    # Early stopping to prevent overfitting..
    early_stop_callback = EarlyStopping(
        monitor="val_edit_distance",  # Metric to monitor for early stopping
        patience=3,  # Stop if no improvement for 3 epochs
        mode="min",  # Lower validation loss is better
    )
    
    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)
    
    # define the trainer 
    trainer = L.Trainer(
        accelerator="gpu",
        #devices=[0],
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        #max_steps=-1,
        limit_val_batches=5,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks= [checkpoint_callback, early_stop_callback])
    trainer.fit(model_module)
    
    
    # Save and zip the model
    save_directory = "/kaggle/working/paligemma_finetuned"
    zip_path = save_model(model_module, save_directory)
    print(f"Model saved at: {zip_path}")
    
    # Inference the fine-tuned model...
    test_example = dataset["test"][0]
    generated_text = infer(test_example, model, processor)
    print(generated_text)

 