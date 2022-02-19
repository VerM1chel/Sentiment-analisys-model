import transformers

train_batch = 10
valid_batch = 5
epochs = 10
max_length = 512

bert_path = './bert-base-cased'
train_data = './imdb.csv'
model_path = './checkpoint.pth'
device = 'cuda'
tokenizer = transformers.BertTokenizer.from_pretrained(bert_path, do_lower_case=True)
