# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 09:52:41 2025

@author: dhars
"""

import re
import json
from torch.utils.data import Dataset
from typing import Any, List, Dict
import random
from datasets import load_dataset
from huggingface_hub import login
from torch.utils.data import DataLoader
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from nltk.metrics.distance import edit_distance
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
import zss  

# Constants
REPO_ID = "google/paligemma-3b-mix-224"
#FINETUNED_MODEL_ID_BASE = "xxxxxxxxxxxxxxxx"
#FINETUNED_MODEL_ID = "xxxxxxxxxxx"
MAX_LENGTH = 512
DATASET_NAME = "naver-clova-ix/cord-v2"
#TOKEN = "xxxxxxxxxxxxxxxxxxxxxx"
PROMPT = "extract JSON."

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
        #Convert an ordered JSON object into a token sequence
        
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
        #Returns one item of the dataset.
        ##Returns:
            #image : the original Receipt image
            #target_sequence : tokenized ground truth sequence
        
        sample = self.dataset[idx]

        # inputs
        image = sample["image"]
        target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        print(target_sequence)
        return image, target_sequence
    
def run_inference(dataset, processor, model, test_custom_dataset):
    total_levenshtein_distance = 0
    total_bleu_score = 0
    total_rouge1 = 0
    total_rouge2 = 0
    total_rougeL = 0
    num_samples = 1 #len(dataset["test"])
    results = []

    for i in range(1):
        test_example = dataset["test"][i]
        test_image = test_example["image"]
        _,target_sequence = test_custom_dataset[i]
        
        inputs = processor(text=PROMPT, images=test_image, return_tensors="pt")
        for k,v in inputs.items():
            print(k,v.shape)
            
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)    
        image_token_index = model.config.image_token_index
        num_image_tokens = len(generated_ids[generated_ids==image_token_index])
        num_text_tokens = len(processor.tokenizer.encode(PROMPT))
        num_prompt_tokens = num_image_tokens + num_text_tokens + 2
        generated_text = processor.batch_decode(generated_ids[:, num_prompt_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print('generated_text',generated_text)
        
        generated_json = token2json(generated_text)
        actual_json = token2json(target_sequence)
        levenshtein_distance = edit_distance(str(generated_json), str(actual_json))
        bleu_score = compute_bleu(str(generated_json), str(actual_json))
        rouge_scores = compute_rouge(str(generated_json), str(actual_json))

        total_levenshtein_distance += levenshtein_distance
        total_bleu_score += bleu_score
        total_rouge1 += rouge_scores["rouge1"].fmeasure
        total_rouge2 += rouge_scores["rouge2"].fmeasure
        total_rougeL += rouge_scores["rougeL"].fmeasure

        results.append({
            "sample": i + 1,
            "generated_json": generated_json,
            "actual_json": actual_json,
            "levenshtein_distance": levenshtein_distance,
            "bleu_score": bleu_score,
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
        })
    
    avg_levenshtein_distance = total_levenshtein_distance / num_samples
    avg_bleu_score = total_bleu_score / num_samples
    avg_rouge1 = total_rouge1 / num_samples
    avg_rouge2 = total_rouge2 / num_samples
    avg_rougeL = total_rougeL / num_samples
    
    return results, avg_levenshtein_distance, avg_bleu_score, avg_rouge1, avg_rouge2, avg_rougeL


def save_results(results, avg_metrics, filename="inference_results.txt"):
    with open(filename, "w") as f:
        for res in results:
            f.write(f"Sample {res['sample']}:\n")
            f.write(f"Generated JSON: {res['generated_json']}\n")
            f.write(f"Actual JSON: {res['actual_json']}\n")
            f.write(f"Levenshtein Distance: {res['levenshtein_distance']}\n")
            f.write(f"BLEU Score: {res['bleu_score']}\n")
            f.write(f"ROUGE Scores: R1={res['rouge1']}, R2={res['rouge2']}, RL={res['rougeL']}\n\n")
        f.write(f"Average Metrics: LD={avg_metrics[0]}, BLEU={avg_metrics[1]}, ROUGE-1={avg_metrics[2]}, ROUGE-2={avg_metrics[3]}, ROUGE-L={avg_metrics[4]}\n")
    print(f"Results saved to {filename}")
    
def token2json(tokens, is_inner_value=False, added_vocab=None):
    
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
            
def compute_bleu(generated, actual):
    return sentence_bleu([actual.split()], generated.split())


def compute_rouge(generated, actual):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return scorer.score(generated, actual)

def main():
    test_custom_dataset = CustomDataset("naver-clova-ix/cord-v2", split="test")
    dataset = load_dataset(DATASET_NAME)
    #login(TOKEN)
    processor = AutoProcessor.from_pretrained(REPO_ID)
    model = PaliGemmaForConditionalGeneration.from_pretrained(REPO_ID)
    results, *avg_metrics = run_inference(dataset, processor, model, test_custom_dataset)
    save_results(results, avg_metrics)
    print(f"Evaluation Metrics: {avg_metrics}")

if __name__ == "__main__":
    test_custom_dataset = CustomDataset("naver-clova-ix/cord-v2", split="test")
    dataset = load_dataset(DATASET_NAME)
    #login(TOKEN)
    processor = AutoProcessor.from_pretrained(REPO_ID)
    model = PaliGemmaForConditionalGeneration.from_pretrained(REPO_ID)
    results, *avg_metrics = run_inference(dataset, processor, model, test_custom_dataset)
    save_results(results, avg_metrics)
    print(f"Evaluation Metrics: {avg_metrics}")
    