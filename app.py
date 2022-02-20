import flask
from flask import Flask
import torch
from model import BertSentimentClassifier
from flask import request

import config

app = Flask(__name__)

def sentence_prediction(sentence):
    sentence = str(sentence)
    sentence = ' '.join(sentence.split())

    input_ids = config.TOKENIZER.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=config.MAX_LENGTH,
            truncation=True,
            pad_to_max_length=True
    )

    ids = input_ids['input_ids']
    mask = input_ids['attention_mask']

    padding_length = config.MAX_LENGTH - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

    ids = ids.to(config.DEVICE, dtype=torch.long)
    mask = mask.to(config.DEVICE, dtype=torch.long)

    outputs = model(ids=ids, attention_mask=mask)
    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]

@app.route("/")
def predict():
    sentence = request.args.get("sentence")
    positive = sentence_prediction(sentence)
    negative = 1 - positive
    response = {}
    response['response'] = {
        "positive": str(positive),
        "negative": str(negative),
        "sentence": str(sentence),
    }
    print(flask.jsonify(response))
    return flask.jsonify(response)

if __name__ == "__main__":
    model = BertSentimentClassifier()
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(config.DEVICE)
    model.eval()
    app.run()
