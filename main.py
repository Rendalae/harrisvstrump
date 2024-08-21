import pandas as pd
from clean import drop_na, drop_certain_names, remove_short_sentences, create_word_sequence
from params import *

def complete_preproc(drop_col = [], na_col = [] , drop_names = [] , min_words= 6, punct_opt=True):

    df = pd.read_csv('data/leg16.csv')
    print('upload achieved')
    print(df.head())

    df = df.drop(columns= DROP_COL) #drop columns with parameters you do not want
    print('columns dropped')
    print(df.head())

    df = drop_na(df, NA_COL) #Removes Nan from specific column
    print('na dropped')
    print(df.head())

    df = drop_certain_names(df, DROP_NAMES) #Removes names you choose
    print('names dropped')
    print(df.head())

    df = remove_short_sentences(df, MIN_WORDS) #Removes sentences with less than n words
    print('sentences dropped')
    print(df.head())

    df = create_word_sequence(df, punct_opt=PUNCT_OPT) #applies the preprocessing, if punct_opt = True includes '!?'
    print(df.head())

    return df
