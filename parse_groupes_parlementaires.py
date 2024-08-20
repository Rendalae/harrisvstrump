import json, os, csv

mandat_folder = './data/raw/groupes/mandat'
organe_folder = './data/raw/groupes/organe'

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


with open("data/acteur-groupe.csv", 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['acteurRef', 'organeRef', 'libelleAbrev', 'libelle']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for json_file in list_json_files(mandat_folder):
        mandat = parse_mandat(json_file)

        if mandat['typeOrgane'] == 'GP':
            groupe_ref = mandat['organeRef']
            groupe=groupes_by_ref[groupe_ref]
            print(mandat)
            writer.writerow({
                'acteurRef': mandat['acteurRef'],
                'organeRef': mandat['organeRef'],
                'libelle': groupe['libelle'],
                'libelleAbrev': groupe['libelleAbrev'],
            })
