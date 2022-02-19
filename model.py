import config
import transformers
import torch.nn as nn


class BertSentimentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768, 1)

    def forward(self, ids, attention_mask):
        out = self.bert(ids, attention_mask=attention_mask)[1]
        out = self.dropout(out)
        output = self.linear(out)
        return output