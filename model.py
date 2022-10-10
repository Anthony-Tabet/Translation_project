import os
# hide TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
from flask import request

class translation:
    def __init__(self, model_path):
        logging.info("translation class initialized")
        self.model = load_model(model_path)
        logging.info("Model is loaded!")
    
    def predict(self, eng_sentence):
        with open('Eng_tokenizer.pickle', 'rb') as handle:
            eng_tokenizer = pickle.load(handle)
        with open('Fr_tokenizer.pickle', 'rb') as handle:
            fr_tokenizer = pickle.load(handle)

        token_to_word = {value: key for key, value in fr_tokenizer.word_index.items()}
        token_to_word[0] = '' 

        en_tokenized = eng_tokenizer.texts_to_sequences(eng_sentence)
        en_padded = pad_sequences(en_tokenized, maxlen = 15, padding = 'post')
        french_translation = np.argmax(self.model.predict(en_padded), axis = -1)[0]
        french_output = ' '.join([token_to_word[tk_wo] for tk_wo in french_translation])
        return french_output
