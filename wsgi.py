from flask import Flask, request, render_template
import pickle
import joblib

import numpy as np
import pandas as pd

import Preprocessing.stopwords as sw
import Preprocessing.tokenAndLemmatiz as tal
import Preprocessing.cvAndTfIdf as cvtf
vectorizer = pickle.load(open('Preprocessing/vectorizer.pkl', 'rb'))

modelOVR = pickle.load(open('ModelsAPI/model_OVR.pkl', 'rb'))

app = Flask(__name__)


@app.route("/")
def home_view():
    return "<h1>Un autre test et un autre commit</h1>"

@app.route("/onevsrest", methods=['POST'])
def predict_tag():
    request_data = request.get_json()

    question = None

    if 'question' in request_data:
        question = request_data['question']
        preprocess_punct = sw.kill_punctuation(question)
        question_clean = sw.Preprocess_listofSentence(preprocess_punct)
        question_list = list(cvtf.sent_to_words(question_clean))
        question_lemmatized = cvtf.lemmatization(question_list, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        vecto = vectorizer.transform(question_lemmatized)
        pred = modelOVR.predict(vecto)
    return ''' Le tag est taratata: {}'''.format(pred)

