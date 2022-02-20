import transformers

TRAIN_BATCH = 10
VALID_BATCH = 5
EPOCHS = 10
MAX_LENGTH = 512

BERT_PATH = './bert-base-cased'
TRAIN_DATA = './imdb.csv'
MODEL_PATH = './checkpoint.pth'
DEVICE = 'cuda'
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH)
