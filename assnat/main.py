from assnat.clean import complete_preproc
import pandas as pd

df = pd.read_csv('data/leg16.csv')
df = complete_preproc(df, na_col = ['Texte'], drop_names = ['Mme la présidente', 'M. le président'], min_words = 6, punct_opt= True)
