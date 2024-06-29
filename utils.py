import torch
from torch.utils.data import Dataset
import torch.nn as nn

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.labels is not None:
            label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        ret_dict = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        if self.labels is not None:
            ret_dict['label'] = torch.tensor(label, dtype=torch.long)
        return ret_dict

class TextClassifier(nn.Module):
    def __init__(self, n_classes, pretrained_model):
        super(TextClassifier, self).__init__()

        self.bert = pretrained_model
        self.bert.config.num_labels = n_classes

        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        res = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = res['pooler_output']
        output = self.drop(pooled_output)
        return self.out(output)
