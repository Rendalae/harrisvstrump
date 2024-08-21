import xml.etree.ElementTree as ET
import csv
import pandas as pd
from utils import list_files

seances_xml_dir = './data/raw/leg15/seances/xml/compteRendu'
seances_csv = './data/leg15-seances.csv'

acteurs_dir = './data/raw/leg15/groupes/xml/acteur'
organes_dir = './data/raw/leg15/groupes/xml/organe'
famille_csv = './data/leg15-acteur-groupe-famille.csv'

all_csv= './data/leg15.csv'

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


def famille_parse_acteur(xml_file):
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
            'acteur_id': acteur_id,
            'organe_id': organe_id,
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
        'uid': uid,
        'code_type': code_type,
        'libelle': libelle,
        'libelle_abrev': libelle_abrev,
        'date_debut': date_debut,
        'date_fin': date_fin
    }

def famille_parse():
    print("Familles parsing")
    acteurs=[]
    for acteurs_file in list_files(acteurs_dir, '.xml'):
        acteurs+=famille_parse_acteur(acteurs_file)
    organes=[]
    for organes_file in list_files(organes_dir, '.xml'):
        organes.append(famille_parse_organe(organes_file))

    acteurs_df = pd.DataFrame.from_records(acteurs)
    organes_df = pd.DataFrame.from_records(organes)
    merged_df = acteurs_df.merge(organes_df, left_on='organe_id', right_on='uid', how='left')
    final_df = merged_df[['acteur_id', 'organe_id', 'date_debut_x', 'date_fin_x', 'qualite', 'libelle', 'code_type','libelle_abrev']]
    final_df = final_df.rename(columns={'libelle': 'groupe_politique', 'libelle_abrev': 'groupe_abrege'})
    final_df.to_csv(famille_csv, index=False, encoding='utf-8')

    print(f"✅ Success : {famille_csv}")


if __name__ == "__main__":
    #seances_parse()
    famille_parse()
