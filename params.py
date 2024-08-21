import os

#pip install python-dotenv
import dotenv, os
dotenv.load_dotenv()

DROP_COL= os.environ.get('DROP_COL')
NA_COL= os.environ.get('NA_COL')
DROP_NAMES = os.environ.get('DROP_NAMES')
MIN_WORDS= int(os.environ.get('MIN_WORDS'))
PUNCT_OPT= bool(os.environ.get('PUNCT_OPT'))
