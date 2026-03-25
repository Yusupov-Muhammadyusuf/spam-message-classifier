import pickle
import torch

from ml.model import SpamClassifier
from utils.preprocessing import clean_text
from utils.tokenizer import Tokenizer

def load_tokenizer():
    with open("tokenizer.pkl", "rb") as file:
        return pickle.load(file)

def load_model(vocab_size):
    model = SpamClassifier(vocab_size)
    model.load_state_dict(torch.load("spam_model.pth"))
    model.eval()

    return model

def predict(text, model, tokenizer, max_len=50):
    text = clean_text(text)
    tokens = tokenizer.encode(text)

    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]

    x = torch.tensor(tokens).unsqueeze(0)

    with torch.no_grad():
        pred = model(x)

    return "SPAM" if pred.item() > 0.5 else "HAM"