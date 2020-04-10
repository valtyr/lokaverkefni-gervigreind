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
