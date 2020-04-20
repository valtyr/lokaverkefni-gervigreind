#! /bin/sh

pip install -r requirements.txt

python app.py

cp model.h5 /artifacts/model.h5
cp tokenizer.pickle /artifacts/tokenizer.pickle
cp training_results.pickle /artifacts/training_results.pickle