#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import json
import pickle
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add


input_img_features =  Input(shape=(2048,))
inp_img1 = Dropout(0.3)(input_img_features)
inp_img2 = Dense(256, activation="relu")(inp_img1)

input_captions = Input(shape=(33,))
inp_cap1 = Embedding(input_dim=1848, output_dim=50, mask_zero=True)(input_captions)
inp_cap2 = Dropout(0.3)(inp_cap1)
inp_cap3 = LSTM(256)(inp_cap2)

decoder1 = add([inp_img2,inp_cap3])
decoder2 = Dense(256,activation="relu")(decoder1)
outputs = Dense(1848, activation="softmax")(decoder2)

model = Model(inputs=[input_img_features, input_captions], outputs=outputs)
model.load_weights("model_weights/model_9.h5")

emodel = ResNet50(weights="imagenet", input_shape=(224,224,3))
new_model = Model(emodel.input,emodel.layers[-2].output)  #prev model GAP layer as output layer

def preprocess_img(img):
    img = image.load_img(img, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    #Normalisation
    img = preprocess_input(img)
    return img
def encode_img(img):
    img = preprocess_img(img)
    feature_vector = new_model.predict(img)
    feature_vector = feature_vector.reshape((1,-1))
    return feature_vector

with open("./storage/word2idx.pkl", "rb") as w2i:
    word2idx = pickle.load(w2i)
with open("./storage/idx2word.pkl", "rb") as i2w:
    idx2word = pickle.load(i2w)

def predict_caption(photo):
    inp_text="<s>"
    for i in range(33):
        sequence = [word2idx[w] for w in inp_text.split() if w in word2idx]
        sequence = pad_sequences([sequence], maxlen=33, padding="post")
        
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx2word[ypred]
        inp_text +=" "+word
        if word == "<e>":
            break
    final_caption = inp_text.split()[1:-1]
    final_caption = " ".join(final_caption)
    return final_caption

def caption(image):
    enc = encode_img(image)
    return predict_caption(enc)

