#! /bin/sh

pip3 install -r requirements.txt

python3 app.py

cp model.h5 /artifacts/model.h5
cp tokenizer.pickle /artifacts/tokenizer.pickle
