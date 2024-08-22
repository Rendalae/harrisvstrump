import pandas as pd
from clean import drop_na, drop_certain_names, remove_short_sentences, create_word_sequence
from params import *

def complete_preproc(drop_col = DROP_COL, na_col = NA_COL , drop_names = DROP_NAMES , min_words= MIN_WORDS, punct_opt=PUNCT_OPT):

    df = pd.read_csv('data/leg16.csv')
    print('Upload achieved')
    print(df.head())

    if drop_col == '' or drop_col == [] or drop_col == None:
       df = df
    else:
        df = df.drop(columns= drop_col) #drop columns with parameters you do not want
    print('Columns dropped')
    print(df.head())

    df = drop_na(df, na_col) #Removes Nan from specific column
    print('NaN dropped')
    print(df.head())

    df = drop_certain_names(df, drop_names) #Removes names you choose
    print('Names dropped')
    print(df.head())

    df = remove_short_sentences(df, min_words) #Removes sentences with less than n words
    print('Short sentences removed')
    print(df.head())

    df = create_word_sequence(df, punct_opt=punct_opt) #applies the preprocessing, if punct_opt = True includes '!?'
    print(df.head())
    print('Preprocessing done!')

    return df
