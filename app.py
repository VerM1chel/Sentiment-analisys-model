import config
import flask
from flask import Flask
import torch
from model import BertSentimentClassifier
from flask import request

app = Flask(__name__)

def sentence_prediction(sentence):
    sentence = str(sentence)
    sentence = ' '.join(sentence.split())

    input_ids = config.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=config.max_length,
            pad_to_max_length=True
        )
    ids = input_ids['input_ids']
    mask = input_ids['attention_mask']

    padding_length = config.max_len - len(ids)
    ids = ids + [0] * padding_length
    mask = mask + [0] * padding_length

    ids = ids.to(config.device, dtype=torch.long)
    mask = mask.to(config.device, dtype=torch.long)

    ids = torch.tensor(ids, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.long)
    outputs = model(ids=ids, mask=mask)
    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]


@app.route('/predict')
def predict():
    sentence = request.args.get('sentence')
    positive = sentence_prediction(sentence)
    negative = 1 - positive
    response = {}
    response["response"] = {
        "positive": str(positive),
        "negative": str(negative),
        "sentence": str(sentence),
    }
    return flask.jsonify(response)


if __name__ == '__main__':
    model = BertSentimentClassifier()
    model.load_state_dict(torch.load(config.model_path))
    model.to(config.device)
    model.eval()
    app.run(host='0.0.0.0', port='9999')







