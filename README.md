# Finetuning_Paligemma_Model

# PaLI-Gemma Fine-Tuning with QLoRA

📌 Project Title: Finetuning_Paligemma_Model
Fine-tuning Paligemma Model on the hugging face dataset.

📖 Table of Contents
  Introduction
  Model
  Results
  Inference
  Future Work

📌 Introduction
This project fine-tunes Paligemma Model to extract text from image. 

🚀 Features:
✔️ Fine-tuning LLaMA using QLoRA for efficiency
✔️ Custom preprocessing and tokenization
✔️ Training on hugging face dataset
✔️ Evaluation & performance metrics

🧠 Model
fine-tuned Paligemma-model using QLoRA to reduce memory usage.

🔧 Hyperparameters:
  Parameter	Value
  Learning Rate	2e-5
  Batch Size	16
  Epochs	3
  
📊 Results
📌 Performance Metrics
  Edit_distance score - base model(), fine-tuned-model()
  rouge_score 
  bleu_score
  
📈 Training Curves
Inference
| Input Image                 | True text     | Predicted text      |
|-----------------------------|---------------|----------------------|
| "image"                     | json          | json                |
 
  

