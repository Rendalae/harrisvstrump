import os

#pip install python-dotenv
import dotenv, os
dotenv.load_dotenv()

DROP_COL= os.environ.get('DROP_COL')
NA_COL= os.environ.get('NA_COL')
DROP_NAMES = os.environ.get('DROP_NAMES')
MIN_WORDS= int(os.environ.get('MIN_WORDS'))
PUNCT_OPT= bool(os.environ.get('PUNCT_OPT'))

FAMILLES_BY_GROUPE = {
    "LAREM": "Centre",
    "HOR": "Centre-droit",
    "UDI": "Centre-droit",
    "NI": "Variable",
    "RN": "Extrême droite",
    "LR": "Droite",
    "UDI_I": "Centre-droit",
    "NG": "Gauche",
    "SOC": "Gauche",
    "RE": "Centre",
    "SRC": "Gauche",
    "DEM": "Centre",
    "MODEM": "Centre",
    "ECOLO": "Gauche",
    "GDR": "Extrême gauche",
    "UDI-AGIR": "Centre-droit",
    "UMP": "Droite",
    "SER": "Gauche",
    "LC": "Centre",
    "LFI-NUPES": "Extrême gauche",
    "LES-REP": "Droite",
    "GDR-NUPES": "Gauche",
    "LIOT": "Centre",
    "UDI-I": "Centre-droit",
    "UDI-A-I": "Centre-droit",
    "R-UMP": "Droite",
    "FI": "Extrême gauche",
    "AGIR-E": "Centre-droit",
    "RRDP": "Centre-gauche",
    "LT": "Centre-droit",
    "EDS": "Centre-gauche"
}
