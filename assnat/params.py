import os

DROP_COLS= []
NA_COLS= []
DROP_NAMES = []
MIN_WORDS= 6
PUNCT_OPT= True

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
