# -*- coding: utf-8 -*-
"""Prediction_dep.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18ka5VO15sqdlzK3OXjWbaK1uNUHRnPDo
"""

!gdown https://drive.google.com/uc?id=1_segDbwreIxKvclmR1-wUUnMWZgL-kqq -O all.zip

import zipfile
!unzip '/content/all.zip'

!pip install -q tensorflow-text
!pip install -q tf-models-official
!pip install -q sklearn
!pip install tensorflow-text

import tensorflow as tf
import tensorflow_text as text
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM,SimpleRNN,GRU,RNN
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from tensorflow.keras import layers

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt

from numpy import array
from numpy import asarray
from numpy import zeros
import tensorflow_hub as hub
import tensorflow as tf
from official.nlp import optimization 
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

def convert2embeding (word):
  with open('/content/content/gdrive/MyDrive/Clasificador/download/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
  X = []
  sentences = list(word)
  #print(sentences)
  for sen in sentences:
    X.append(preprocess_text(sen))
  #print(X)
  token_word=tokenizer.texts_to_sequences(X[0])
  #print(token_word)
  final=[]
  for tok in token_word:
    try:
      final.append(tok[0])
    except:
      final.append(0)
  #print(final)
  maxlen = 32
  final_word = pad_sequences([final], padding='post', maxlen=maxlen)
  #print(final_word)
  return final_word
word2='Beatiful world baby'
x=convert2embeding ([word2])
print(x)

glove_dataset_dep='/content/content/gdrive/MyDrive/Clasificador/download/glove_dataset_hugging.h5'
fasttext_dataset_dep='/content/content/gdrive/MyDrive/Clasificador/download/Fasttext_dataset_hugging.h5'
glove = keras.models.load_model(glove_dataset_dep)
fasttext = keras.models.load_model(fasttext_dataset_dep)

def predictions (word2pred,glove_model,fasttext_model):
  #print("1")
  convertion=convert2embeding([word2pred])
  #print("2")
  glove_pred = glove_model.predict(convertion)
  #print("3")
  fastext_pred = fasttext_model.predict(convertion)
  #print("4")
  final_pred=[]
  #print("5")
  for pred in range(len(glove_pred)):
    final_pred.append((glove_pred[pred]+fastext_pred[pred])/2)
    #print("final_pred = ",final_pred)
    #print("glove_pred[pred] = ",glove_pred[pred])
    #print("fastext_pred[pred] = ",fastext_pred[pred])
  return final_pred[0]

frase='hello beautiful'
x=predictions (frase,glove,fasttext)
print(x)