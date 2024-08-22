import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import string
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import text_to_word_sequence


def drop_na(df, column): #Removes Nan from specific column
    if column == '' or column == [] or column == None:
        df = df
    else:
        df = df.dropna(subset = column)
    return df

def drop_certain_names(df,names): #Removes names you choose
    if names == '' or names == [] or names == None:
        df = df
    else:
        df = df[~df['Nom Orateur'].isin(names)]
    return df

def remove_short_sentences(df, n): #Removes sentences with less than n words
    def word_count(sentence):
        return len(sentence.split())
    df['Texte'] = df['Texte'].astype(str)
    df = df[df['Texte'].apply(word_count) >= n]
    return df

# Turn dates into years

def create_word_sequence(df, punct_opt = True, text_to_sequence = False): #applies the preprocessing, if punct_opt = True keeps '!?'

    def preprocessing(sentence, punct_option = True):
        # Removing whitespaces
        sentence = sentence.strip()
        # Lowercasing
        sentence = sentence.lower()
        # Removing punctuation
        if punct_option == True:
            for punctuation in string.punctuation.replace('?','').replace('!',''):
                sentence = sentence.replace(punctuation, '')
        else:
            for punctuation in string.punctuation:
                sentence = sentence.replace(punctuation, '')
        return sentence

    df['Texte'] = df['Texte'].apply(preprocessing)
    print(type(df['Texte']))
    if text_to_sequence:
        print('text_to_sequence')
        df['Texte'] = df['Texte'].apply(text_to_word_sequence,filters=string.punctuation.replace('?','').replace('!',''), lower=True, split=' ')
    return df
