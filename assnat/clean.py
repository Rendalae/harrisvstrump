import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import string
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from assnat.params import *


def complete_preproc(df, drop_col = DROP_COLS, simplify_fam = SIMPLIFY_FAM, drop_fam = DROP_FAM, na_col = NA_COLS , drop_names = DROP_NAMES , min_words= MIN_WORDS, punct_opt=PUNCT_OPT):
    if len(drop_col)>0:
        df = df.drop(columns= drop_col) #drop columns with parameters you do not want
    print('Columns dropped')
    #print(df.head())

    if simplify_fam:
        famille_broad = {
        'Centre': 'Centre',
        'Centre-droit': 'Centre',
        'Centre-gauche': 'Centre',
        'Droite': 'Droite',
        'Extrême droite': 'Droite',
        'Gauche': 'Gauche',
        'Extrême gauche': 'Gauche',
        'Variable': 'Variable'
        }

        print(df)
        df['famille'] = df['famille'].apply(lambda x: famille_broad.get(x, x))
        print('families simplified')

    df = drop_certain_families(df, drop_fam)
    print('family dropped')
    #print(df.head())

    df = drop_na(df, na_col) #Removes Nan from specific column
    print('NaN dropped')
    #print(df.head())

    df = drop_certain_names(df, drop_names) #Removes names you choose
    print('Names dropped')
    #print(df.head())

    df = remove_short_sentences(df, min_words) #Removes sentences with less than n words
    print('Short sentences removed')
    #print(df.head())

    df = create_word_sequence(df, punct_opt=punct_opt) #applies the preprocessing, if punct_opt = True includes '!?'
    #print(df.head())
    print('Preprocessing done!')

    return df





def drop_certain_families(df,family): # Removes families you choose
    if len(family)>0:
        df = df[~df['famille'].isin(family)]
    return df

def drop_na(df, column): #Removes Nan from specific column
    if len(column)>0:
        df = df.dropna(subset = column)
    return df

def drop_certain_names(df,names): #Removes names you choose
    if len(names)>0:
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
    if text_to_sequence:
        df['Texte'] = df['Texte'].apply(text_to_word_sequence,filters=string.punctuation.replace('?','').replace('!',''), lower=True, split=' ')
    return df
