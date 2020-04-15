from data import load_data
from tokenizer import Tokenizer
from model import Model

training_data, validation_data = load_data()

tokenizer = Tokenizer()
tokenizer.fit(training_data)

tokenized_data = tokenizer.tokenize(training_data)
tokenized_validation_data = tokenizer.tokenize(validation_data)

model = Model()
model.build(tokenized_data)

history = model.fit(tokenized_data)
print('\nhistory dict:', history.history)

print('\n# Evaluate on test data')
results = model.evaluate(tokenized_validation_data)
print('test loss, test acc:', results)


def predict_sentence(sentence: str):
    tokenized_sentence = tokenizer.tokenize_sentence(sentence)
    return model.classify(tokenized_sentence)


print(
    predict_sentence("I love you son you are the prettiest boy and most kind"))

model.save()
tokenizer.save()