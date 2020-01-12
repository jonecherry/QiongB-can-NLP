import sys, os, re, csv, codecs, numpy as np, pandas as pd
# =================Keras==============
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Conv1D, Conv2D, Embedding, Dropout, Activation, GlobalMaxPooling1D
from keras.layers import Bidirectional, MaxPooling1D, MaxPooling2D, Reshape, Flatten, concatenate, BatchNormalization
from keras.layers import GlobalAveragePooling1D, SpatialDropout1D, GRU, LSTM
from keras import backend as K
from keras.engine import Layer
from keras.layers import MaxPool2D, Concatenate
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers, backend

# =================nltk===============
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

