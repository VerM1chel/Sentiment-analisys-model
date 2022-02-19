import config
import dataset
from model import BertSentimentClassifier
import engine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import torch
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import re


def clean_text(texts):
    for i, text in enumerate(texts):
        texts[i] = re.sub(r"<br />", "", text)
        texts[i] = re.sub("'ll", ' will', text)
        texts[i] = re.sub("'ve", ' have', text)
    return texts


def run():
    data = pd.read_csv(config.train_data).fillna('None')
    data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    data['review'] = clean_text(data['review'])

    train, valid = train_test_split(data, test_size=0.1, stratify=data['sentiment'].values, random_state=99)
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)

    trainDataset = dataset.BertDataset(train['review'].values, train['sentiment'].values)
    trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=config.train_batch, num_workers=4)

    validDataset = dataset.BertDataset(valid['review'].values, valid['sentiment'].values)
    validDataloader = torch.utils.data.DataLoader(validDataset, batch_size=config.valid_batch, num_workers=1)

    device = torch.device('cuda')
    model = BertSentimentClassifier()
    model.to(device)

    parameters = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [param for n, param in parameters if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.001},
        {'params': [param for n, param in parameters if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}  # !
    ]

    optimizer = AdamW(optimizer_parameters, lr=5e-5)

    num_training_steps = int(len(trainDataset) / config.train_batch * config.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0,
                                                num_training_steps=num_training_steps)

    best_accuracy = 0
    for epoch in range(config.epochs):
        print(f'epoch_{epoch + 1}')
        engine.train_fn(trainDataloader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(validDataloader, model, device)
        outputs = np.array(outputs) > 0.5
        accuracy = metrics.accuracy_score(outputs, targets)
        print(f'Accuracy score = {accuracy}')
        if accuracy > best_accuracy:
            print('nice')
            best_accuracy = accuracy
            torch.save(model.state_dict(), config.model_path)
            torch.save(optimizer.state_dict(), '/kaggle/working/opt.pth')  # !
            torch.save(scheduler.state_dict(), '/kaggle/working/shed.pth')  # !


if __name__ == '__main__':
    run()