import os

#pip install python-dotenv
import dotenv, os
dotenv.load_dotenv()

DROP_COL= '' #os.environ.get('DROP_COL')
NA_COL= ['Texte'] #os.environ.get('NA_COL')
DROP_NAMES = ['M. le président', 'Mme la présidente'] #os.environ.get('DROP_NAMES')
MIN_WORDS= 6 #int(os.environ.get('MIN_WORDS'))
PUNCT_OPT= True #bool(os.environ.get('PUNCT_OPT'))
