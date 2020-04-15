from model import Model
from tokenizer import Tokenizer

model = Model.load()
tokenizer = Tokenizer.load()


def predict_sentence(sentence: str):
    tokenized_sentence = tokenizer.tokenize_sentence(sentence)
    return model.classify(tokenized_sentence)


from IPython import embed
embed()
