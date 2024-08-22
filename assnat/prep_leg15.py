import xml.etree.ElementTree as ET
import csv
import pandas as pd
from assnat.params import FAMILLES_BY_GROUPE
from assnat.utils import append_to_dict_key, list_files

seances_xml_dir = './data/raw/leg15/seances/xml/compteRendu'
seances_csv = './data/leg15-seances.csv'

acteurs_dir = './data/raw/leg15/groupes/xml/acteur'
organes_dir = './data/raw/leg15/groupes/xml/organe'
famille_csv = './data/leg15-acteur-groupe-famille.csv'

all_csv= './data/leg15-without-family.csv'

ns = {'ns': 'http://schemas.assemblee-nationale.fr/referentiel'}


def seances_parse():
    print("Seances parsing")
    # Initialiser la liste des données
    data = []

    # Parcourir tous les fichiers XML dans le répertoire
    for xml_file in list_files(seances_xml_dir, '.xml'):
        print(xml_file, end='\r')
        tree = ET.parse(xml_file)
        root = tree.getroot()

        date_seance = root.find('ns:metadonnees/ns:dateSeanceJour', ns).text if root.find('ns:metadonnees/ns:dateSeanceJour', ns) is not None else "Date inconnue"
        id_seance = root.find('ns:metadonnees/ns:numSeance', ns).text if root.find('ns:metadonnees/ns:numSeance', ns) is not None else "ID inconnu"
        theme_seance = root.find('ns:metadonnees/ns:sommaire/ns:sommaire1/ns:titreStruct/ns:intitule', ns).text if root.find('ns:metadonnees/ns:sommaire/ns:sommaire1/ns:titreStruct/ns:intitule', ns) is not None else "Thème inconnu"
        id_session = root.find('ns:metadonnees/ns:session', ns).text if root.find('ns:metadonnees/ns:session', ns) is not None else "Session inconnue"

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


def famille_parse_acteur_mandats(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extraire l'ID de l'acteur
    acteur_id = root.find('ns:uid', ns).text if root.find('ns:uid', ns) is not None else "ID inconnu"

    # Extraire les informations des mandats
    acteur_mandats = []
    for mandat in root.findall('.//ns:mandat', ns):
        organe_id = mandat.find('.//ns:organes/ns:organeRef', ns).text if mandat.find('.//ns:organes/ns:organeRef', ns) is not None else "Organe inconnu"
        date_debut = mandat.find('ns:dateDebut', ns).text if mandat.find('ns:dateDebut', ns) is not None else "Date inconnue"
        date_fin = mandat.find('ns:dateFin', ns).text if mandat.find('ns:dateFin', ns) is not None else "Date inconnue"
        qualite = mandat.find('.//ns:infosQualite/ns:libQualite', ns).text if mandat.find('.//ns:infosQualite/ns:libQualite', ns) is not None else "Qualité inconnue"

        # Ajouter les données extraites à la liste
        acteur_mandats.append( {
            'acteurRef': acteur_id,
            'organeRef': organe_id,
            'date_debut': date_debut,
            'date_fin': date_fin,
            'qualite': qualite
        })
    return acteur_mandats

def famille_parse_organe(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extraire les informations du fichier
    uid = root.find('ns:uid', ns).text if root.find('ns:uid', ns) is not None else "UID inconnu"
    code_type = root.find('ns:codeType', ns).text if root.find('ns:codeType', ns) is not None else "Code Type inconnu"
    libelle = root.find('ns:libelle', ns).text if root.find('ns:libelle', ns) is not None else "Libellé inconnu"
    libelle_abrev = root.find('ns:libelleAbrev', ns).text if root.find('ns:libelleAbrev', ns) is not None else "Libellé Abrégé inconnu"
    date_debut = root.find('.//ns:viMoDe/ns:dateDebut', ns).text if root.find('.//ns:viMoDe/ns:dateDebut', ns) is not None else "Date Début inconnue"
    date_fin = root.find('.//ns:viMoDe/ns:dateFin', ns).text if root.find('.//ns:viMoDe/ns:dateFin', ns) is not None else "Date Fin inconnue"

    # Ajouter les données extraites à la liste
    return {
        'organeRef': uid,
        'typeOrgane': code_type,
        'libelle': libelle,
        'libelle_abrev': libelle_abrev,
        'date_debut': date_debut,
        'date_fin': date_fin
    }

# Crée le dictionnaire des groupes parlementaires
# default is 'GP' for groupes parlementaires
# mais le parame 'ASSEMBLEE' est aussi possible pour avoir la placeHemicycle
def famille_create_organe_dic(typeOrgane = 'GP'):
    groupe_by_id = {}
    for json_file in list_files(organes_dir, '.xml'):
        print(json_file, end='\r')
        organe = famille_parse_organe(json_file)
        if organe['typeOrgane'] == typeOrgane:
            organeRef = organe['organeRef']
            groupe_by_id[organeRef] = organe
    return groupe_by_id


# Crée le fichier CSV des acteurs avec groupe et famille
def famille_parse():
    print("Family parsing")
    organes = famille_create_organe_dic('GP')
    acteur_groupes={}
    for xml_file in list_files(acteurs_dir,'.xml'):
        for mandat in famille_parse_acteur_mandats(xml_file):
            organeRef = mandat['organeRef']
            acteur_ref = mandat['acteurRef']
            if not organeRef in organes:
                continue
            organe = organes[organeRef]
            groupe_code=organe['libelle_abrev']
            append_to_dict_key(acteur_groupes, acteur_ref, groupe_code)

    groupe_famille_by_acteur=[]
    for acteur, organes in acteur_groupes.items():
        final_groupe='NI'
        for organe in organes:
            if(organe!='NI'):
                final_groupe=organe
                # On prend le premier groupe qui n'est pas NI !!!!!
                break
        if final_groupe in FAMILLES_BY_GROUPE:
            famille = FAMILLES_BY_GROUPE[final_groupe]
        else:
            print(f"⚠️⚠️⚠️ Missing group {final_groupe}")
            famille = "NA"
        groupe_famille_by_acteur.append( {'ID Orateur': acteur[2:],'groupe':organe, 'famille': famille})

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
    #seances_parse()
    #famille_parse()
    all_parse()
