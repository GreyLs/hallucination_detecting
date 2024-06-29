import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from tqdm import tqdm
import os
import pickle

from utils import TextDataset, TextClassifier

def main():
    # Загрузка токенизатора и модели из файла
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    with open('model.pkl', 'rb') as f:
        pretrained_model = pickle.load(f)

    # Создаем тестовый DataLoader 
    df = pd.read_csv("data/test.csv")
    texts = "summary: " + df["summary"] + " | question: " + df["question"] + " | answer: " + df["answer"]
    labels = df["is_hallucination"]

    # Create dataset
    dataset = TextDataset(texts, labels, tokenizer)
    
    # Create dataloader
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    device = torch.device("mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = pretrained_model.to(device)
    model = TextClassifier(pretrained_model=pretrained_model)
    model = model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            predictions.extend(preds)

    # Создание файла submission.csv
    submission_df = pd.DataFrame({'id': df["line_id"], 'is_hallucination': predictions})
    submission_df.to_csv('data/submission.csv', index=False)

if __name__ == "__main__":
    main()
