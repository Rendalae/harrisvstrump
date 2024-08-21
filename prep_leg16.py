import xml.etree.ElementTree as ET
import json, csv
import pandas as pd
from utils import append_to_dict_key, list_files

seances_xml_dir = './data/raw/leg16/seances/xml/compteRendu'
seances_csv = './data/leg16-seances.csv'

votes_xml_dir = './data/raw/leg16/votes/xml'
votes_csv = './data/leg16-votes.json'

mandats_dir = './data/raw/leg16/groupes/mandat'
organes_dir = './data/raw/leg16/groupes/organe'
famille_csv = './data/leg16-acteur-groupe-famille.csv'
mandats_csv = './data/leg16-mandats.csv'

all_csv= './data/leg16.csv'

ns = {'ns': 'http://schemas.assemblee-nationale.fr/referentiel'}

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

def seances_parse():
    print("Seances parsing")
    # Initialiser la liste des données
    data = []

    # Parcourir tous les fichiers XML dans le répertoire
    for xml_file in list_files(seances_xml_dir, '.xml'):
        print(xml_file, end='\r')
        # Charger le fichier XML
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extraire les métadonnées de la séance
        date_seance = root.find('ns:metadonnees/ns:dateSeanceJour', ns).text if root.find('ns:metadonnees/ns:dateSeanceJour', ns) is not None else "Date inconnue"
        id_seance = root.find('ns:seanceRef', ns).text if root.find('ns:seanceRef', ns) is not None else "ID inconnu"
        theme_seance = root.find('ns:metadonnees/ns:sommaire/ns:sommaire1/ns:titreStruct/ns:intitule', ns).text if root.find('ns:metadonnees/ns:sommaire/ns:sommaire1/ns:titreStruct/ns:intitule', ns) is not None else "Thème inconnu"
        id_session = root.find('ns:sessionRef', ns).text if root.find('ns:sessionRef', ns) is not None else "ID inconnu"

        # Extraire les informations des orateurs
        for paragraphe in root.findall('.//ns:paragraphe', ns):
            orateurs = paragraphe.find('ns:orateurs', ns)
            if orateurs is not None:
                for orateur in orateurs.findall('ns:orateur', ns):
                    nom_orateur = orateur.find('ns:nom', ns).text if orateur.find('ns:nom', ns) is not None else "Nom inconnu"
                    id_orateur = orateur.find('ns:id', ns).text if orateur.find('ns:id', ns) is not None else "ID inconnu"
                    texte = paragraphe.find('ns:texte', ns).text if paragraphe.find('ns:texte', ns) is not None else ""
                    grammaire = paragraphe.get('code_grammaire', "")
                    data.append([nom_orateur, id_orateur, texte, theme_seance, id_seance, date_seance, id_session, grammaire])

    # Écrire les données dans un fichier CSV concaténé
    with open(seances_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Nom Orateur', 'ID Orateur', 'Texte', 'Thème Séance', 'ID Séance', 'Date Séance', 'ID Session', 'grammaire'])
        writer.writerows(data)

    print(f"✅ Success : {seances_csv}")

def votes_parse():
    print("Votes parsing")
    all_votes_data = []

    for xml_file in list_files(votes_xml_dir, '.xml'):
        print(xml_file, end='\r')

        tree = ET.parse(xml_file)
        root = tree.getroot()

        titre = root.find('.//ns:titre', ns).text
        for group in root.findall('.//ns:groupes/ns:groupe', ns):
            group_id = group.find('ns:organeRef', ns).text

            pours=[]
            for votant in group.findall('.//ns:decompteNominatif/ns:pours/ns:votant', ns):
                acteur_id = votant.find('ns:acteurRef', ns).text
                pours.append(acteur_id)

            contres=[]
            for votant in group.findall('.//ns:decompteNominatif/ns:contres/ns:votant', ns):
                acteur_id = votant.find('ns:acteurRef', ns).text
                contres.append(acteur_id)

            all_votes_data.append({
                'titre': titre,
                'groupe': group_id,
                'pours': pours,
                'contres': contres
            })

    with open(votes_csv, 'w', newline='', encoding='utf-8') as csvfile:
        json.dump(all_votes_data, csvfile, ensure_ascii=False, indent=4)
    print(f"✅ Success : {votes_csv}")



# Parsing d'un fichiers JSON de mandat
def famille_parse_mandat(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extraire les valeurs
    acteur_id = data['mandat']['acteurRef']
    type_organe = data['mandat']['typeOrgane']
    organe_id = data['mandat']['organes']['organeRef']
    if 'mandature' in data['mandat']:
        placeHemicycle = data['mandat']['mandature']['placeHemicycle'] #TODO
    else:
        placeHemicycle = None

    return {
        "acteurRef": acteur_id,
        "typeOrgane": type_organe,
        "organeRef": organe_id,
        "placeHemicycle": placeHemicycle
    }

# Parsing d'un fichiers JSON d'organe
def famille_parse_organe(json_path):
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
# default is 'GP' for groupes parlementaires
# mais le parame 'ASSEMBLEE' est aussi possible pour avoir la placeHemicycle
def famille_create_organe_dic(typeOrgane = 'GP'):
    groupe_by_id = {}
    for json_file in list_files(organes_dir, '.json'):
        print(json_file, end='\r')
        organe = famille_parse_organe(json_file)
        if organe['typeOrgane'] == typeOrgane:
            organeRef = organe['organeRef']
            groupe_by_id[organeRef] = organe
    return groupe_by_id

# Export the Mandats as CSV for further analysis
def export_mandats():
    print("Exporting mandats")
    with open(mandats_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['typeOrgane', 'organeRef', 'acteurRef','placeHemicycle']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for json_file in list_files(mandats_dir,'.json'):
            print(json_file, end='\r')
            mandat = famille_parse_mandat(json_file)
            if mandat['typeOrgane'] != 'ASSEMBLEE':
                continue
            writer.writerow({'typeOrgane':mandat['typeOrgane'],
                             'organeRef':mandat['organeRef'],
                             'acteurRef':mandat['acteurRef'],
                             'placeHemicycle': mandat.get('placeHemicycle','')
                             })
    print(f"✅ Success : {mandats_csv}")

# Crée le fichier CSV des acteurs avec groupe et famille
def famille_parse():
    print("Family parsing")
    groupes = famille_create_organe_dic('GP')
    acteur_groupes={}
    for json_file in list_files(mandats_dir,'.json'):
        print(json_file, end='\r')
        mandat = famille_parse_mandat(json_file)
        if mandat['typeOrgane'] == 'GP':
            groupe_id = mandat['organeRef']
            groupe_code=groupes[groupe_id]['libelleAbrev']
            acteur_id = mandat['acteurRef']
            append_to_dict_key(acteur_groupes, acteur_id, groupe_code)

    groupe_famille_by_acteur=[]
    for acteur, groupes in acteur_groupes.items():
        final_groupe='NI'
        for groupe in groupes:
            if(groupe!='NI'):
                final_groupe=groupe
                # On prend le premier groupe qui n'est pas NI !!!!!
                break
        if not final_groupe in famille_by_groupe:
            print(f"⚠️⚠️⚠️ Missing group {final_groupe}")
        famille = famille_by_groupe[final_groupe]
        groupe_famille_by_acteur.append( {'ID Orateur': acteur[2:] ,'groupe':groupe, 'famille': famille})

    with open(famille_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['ID Orateur', 'groupe', 'famille']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in groupe_famille_by_acteur:
            writer.writerow(entry)
    print(f"✅ Success : {famille_csv}")


def all_parse():
    print("Packing all data")
    seances = pd.read_csv(seances_csv)
    seances["ID Orateur"]=seances["ID Orateur"].astype(str)
    acteur_famille=pd.read_csv(famille_csv)
    acteur_famille=acteur_famille[["ID Orateur", "famille"]]
    acteur_famille=acteur_famille.astype(str)
    all= seances.merge(acteur_famille, on="ID Orateur", how="left")
    all.to_csv(all_csv, index=False)
    print(f"✅ Success : {all_csv}")


if __name__ == "__main__":
    seances_parse()
    votes_parse()
    famille_parse()
    all_parse()
    export_mandats()
