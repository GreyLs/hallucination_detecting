import torch
import pandas as pd
from torch.utils.data import DataLoader
from utils import TextClassifier
import pickle  

def make_predictions(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())

            del input_ids, attention_mask, outputs
            torch.cuda.empty_cache()

    return predictions

def main():
    test_df = pd.read_csv("data/test.csv")
    texts = "summary: " + test_df["summary"] + " | question: " + test_df["question"] + " | answer: " + test_df["answer"]

    # Загрузка токенизатора из файла
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    encoded_data = tokenizer(
        list(texts), 
        add_special_tokens=True, 
        max_length=512, 
        padding='max_length', 
        truncation=True, 
        return_attention_mask=True,
        return_tensors='pt'
    )

    data_loader = DataLoader(
        [{'input_ids': input_ids, 'attention_mask': attention_mask} 
         for input_ids, attention_mask in 
         zip(encoded_data['input_ids'], encoded_data['attention_mask'])], 
        batch_size=2, 
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
    # Загрузка модели из файла
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    model = model.to(device)

    model = TextClassifier(n_classes=2, pretrained_model=model) 

    predictions = make_predictions(model, data_loader, device)

    submission_df = pd.DataFrame({
        "line_id": test_df["line_id"],
        "is_hallucination": predictions
    })
    submission_df.to_csv("data/submission.csv", index=False)

if __name__ == "__main__":
    main()
