import json, os, csv

mandat_folder = './data/raw/groupes/mandat'
organe_folder = './data/raw/groupes/organe'

famille_by_groupe = {
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
    "GDR": "Gauche",
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
    "LT": "Centre",
    "EDS": "Centre-gauche"
}

# Liste des fichiers JSON dans un folder
def list_json_files(folder):
    json_files=[]
    for file_name in os.listdir(folder):
            if not file_name.endswith('.json'):
                continue
            json_path = os.path.join(folder, file_name)
            json_files.append(json_path)
    return json_files

# Parsing d'un fichiers JSON de mandat
def parse_mandat(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extraire les valeurs
    acteur_ref = data['mandat']['acteurRef']
    type_organe = data['mandat']['typeOrgane']
    organe_ref = data['mandat']['organes']['organeRef']

    return {
        "acteurRef": acteur_ref,
        "typeOrgane": type_organe,
        "organeRef": organe_ref
    }

# Parsing d'un fichiers JSON d'organe
def parse_organe(json_path):
    # Ouvrir le fichier JSON et charger les données
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extraire les valeurs
    uid = data['organe']['uid']
    typeOrgane = data['organe']['codeType']
    libelle = data['organe']['libelle']
    libelle_abrev = data['organe']['libelleAbrev']

    return {
        "organeRef": uid,
        "typeOrgane": typeOrgane,
        "libelle": libelle,
        "libelleAbrev": libelle_abrev
    }

# Crée le dictionnaire des groupes parlementaires
def parse_groupes_by_ref():
    groupe_by_ref = {}
    for json_file in list_json_files(organe_folder):
        organe = parse_organe(json_file)
        if organe['typeOrgane'] == 'GP':
            organeRef = organe['organeRef']
            groupe_by_ref[organeRef] = organe
    return groupe_by_ref


groupes_by_ref = parse_groupes_by_ref()

def append(dict, key, val):
    if key not in dict:
        dict[key] = []
    dict[key].append(val)


groupes_by_acteur={}
for json_file in list_json_files(mandat_folder):
        mandat = parse_mandat(json_file)
        if mandat['typeOrgane'] == 'GP':
            groupe_ref = mandat['organeRef']
            groupe_code=groupes_by_ref[groupe_ref]['libelleAbrev']
            acteur_ref = mandat['acteurRef']
            append(groupes_by_acteur, acteur_ref, groupe_code)

groupe_famille_by_acteur=[]
for acteur, groupes in groupes_by_acteur.items():
    final_groupe='NI'
    for groupe in groupes:
        if(groupe!='NI'):
            final_groupe=groupe
            # On prend le premier groupe qui n'est pas NI !!!!!
            break
    if not final_groupe in famille_by_groupe:
        print(f"Missing group {final_groupe}")
    famille = famille_by_groupe[final_groupe]
    groupe_famille_by_acteur.append( {'ID Orateur': acteur[2:] ,'groupe':groupe, 'famille': famille})

print(groupe_famille_by_acteur)


with open("data/acteur-groupe.csv", 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['ID Orateur', 'groupe', 'famille']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for entry in groupe_famille_by_acteur:
        writer.writerow(entry)
