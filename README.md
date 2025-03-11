# Finetuning_Paligemma_Model

# PaLI-Gemma Fine-Tuning with QLoRA

📌 Project Title: Finetuning Paligemma Model

  Fine-tuning Paligemma Model on the hugging face dataset to extract text from image.

📖 Table of Contents

  - Introduction
  - Features
  - Model
  - Hyperparameters
  - Results

📌 Introduction

  This project fine-tunes Paligemma Model to extract text from image. 

🚀 Features

  ✔️ Fine-tuning Paligemma Model using QLoRA for efficiency

  ✔️ Custom preprocessing and tokenization

  ✔️ Training on hugging face dataset

  ✔️ Evaluation & performance metrics

🧠 Model
  
   Fine-tuned Paligemma-model using QLoRA to reduce memory usage.

🔧 Hyperparameters

| Parameter      | Value   |
|----------------|----------|
| **Learning Rate** | 100.4 |
| **Batch Size**    | 0.6   |
| **Epochs**        | 30    |

  
📊 Results
    
**Model Evaluation Metrics**

| Metrics               | Base Model | Fine-tuned Model|
|-----------------------|-----------|------------------|
| **Edit Distance Score** | 100.4    | 64.3            |
| **ROUGE Score**        | 0.6       | 0.7             |
| **BLEU Score**         | 0.2       | 0.4             |

**Inference Results**

| Input Image | True Text | Predicted Text |
|-------------|-----------|-----------------|
| image | `{'total': {'total_price': '91000', 'cashprice': '91000'}, 'menu': [{'price': '17500', 'nm': 'J.STB PROMO'}, {'price': '46000', 'nm': 'Y.B.BAT'}, {'price': '27500', 'nm': 'Y.BASO PROM'}]}` | `{'total': {'total_price': '91000', 'cashprice': '91000'}, 'menu': [{'price': '17500', 'nm': 'J.SIB PROMO'}, {'price': '45000', 'nm': 'V.B.BAT'}, {'price': '27500', 'nm': 'Y.BASC PROM'}]}` |
