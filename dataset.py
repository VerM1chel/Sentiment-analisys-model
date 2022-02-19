import config
import torch

class BertDataset:
    def __init__(self, text, targets):
        self.text = text
        self.targets = targets
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = str(self.text[idx])
        text = ' '.join(text.split())

        input_ids = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True
        )

        ids = input_ids['input_ids']
        mask = input_ids['attention_mask']

        padding_length = self.max_length - len(ids)
        ids = ids + [0] * padding_length
        mask = mask + [0] * padding_length

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[idx], dtype=torch.float)
        }