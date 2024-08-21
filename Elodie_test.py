directory = '/Users/eloa/Downloads/xml-4/acteur'  # Remplacez par le chemin de votre répertoire

# Initialiser une liste pour stocker les données extraites de chaque fichier
data = []

# Espaces de noms XML
ns = {'ns': 'http://schemas.assemblee-nationale.fr/referentiel'}

# Parcourir chaque fichier XML dans le répertoire
for filename in os.listdir(directory):
    if filename.endswith('.xml'):
        # Charger le fichier XML
        xml_file = os.path.join(directory, filename)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extraire l'ID de l'acteur
        acteur_id = root.find('ns:uid', ns).text if root.find('ns:uid', ns) is not None else "ID inconnu"

        # Extraire les informations des mandats
        for mandat in root.findall('.//ns:mandat', ns):
            organe_id = mandat.find('.//ns:organes/ns:organeRef', ns).text if mandat.find('.//ns:organes/ns:organeRef', ns) is not None else "Organe inconnu"
            date_debut = mandat.find('ns:dateDebut', ns).text if mandat.find('ns:dateDebut', ns) is not None else "Date inconnue"
            date_fin = mandat.find('ns:dateFin', ns).text if mandat.find('ns:dateFin', ns) is not None else "Date inconnue"
            qualite = mandat.find('.//ns:infosQualite/ns:libQualite', ns).text if mandat.find('.//ns:infosQualite/ns:libQualite', ns) is not None else "Qualité inconnue"

            # Ajouter les données extraites à la liste
            data.append({
                'acteur_id': acteur_id,
                'organe_id': organe_id,
                'date_debut': date_debut,
                'date_fin': date_fin,
                'qualite': qualite
            })

# Convertir la liste de données en DataFrame
df = pd.DataFrame(data)

# Sauvegarder le DataFrame final dans un fichier CSV
output_csv = '/Users/eloa/Downloads/xml-4/acteurs_mandats_concat.csv'  # Remplacez par le chemin de votre fichier de sortie
df.to_csv(output_csv, index=False, encoding='utf-8')

print(f"Fichier CSV concaténé généré avec succès : {output_csv}")

# Répertoire contenant les fichiers XML
directory = '/Users/eloa/Downloads/xml-4/organe'  # Remplacez par le chemin de votre répertoire

# Initialiser une liste pour stocker les données extraites de chaque fichier
data = []

# Espaces de noms XML
ns = {'ns': 'http://schemas.assemblee-nationale.fr/referentiel'}

# Parcourir chaque fichier XML dans le répertoire
for filename in os.listdir(directory):
    if filename.endswith('.xml'):
        # Charger le fichier XML
        xml_file = os.path.join(directory, filename)
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
        data.append({
            'uid': uid,
            'code_type': code_type,
            'libelle': libelle,
            'libelle_abrev': libelle_abrev,
            'date_debut': date_debut,
            'date_fin': date_fin
        })

# Convertir la liste de données en DataFrame
df = pd.DataFrame(data)

# Sauvegarder le DataFrame final dans un fichier CSV
output_csv = '/Users/eloa/Downloads/xml-4/organes_concat.csv'
df.to_csv(output_csv, index=False, encoding='utf-8')

print(f"Fichier CSV concaténé généré avec succès : {output_csv}")

import pandas as pd

# Chemins des fichiers CSV
acteurs_file = '/Users/eloa/Downloads/xml-4/acteurs_mandats_concat.csv'
organes_file = '/Users/eloa/Downloads/xml-4/organes_concat.csv'

# Charger les fichiers CSV dans des DataFrames
acteurs_df = pd.read_csv(acteurs_file)
organes_df = pd.read_csv(organes_file)

# Fusionner les deux DataFrames sur la colonne 'organe_id'
merged_df = acteurs_df.merge(organes_df, left_on='organe_id', right_on='uid', how='left')

# Sélectionner les colonnes pertinentes
final_df = merged_df[['acteur_id', 'organe_id', 'date_debut_x', 'date_fin_x', 'qualite', 'libelle', 'code_type','libelle_abrev']]

# Renommer les colonnes si nécessaire
final_df.rename(columns={'libelle': 'groupe_politique', 'libelle_abrev': 'groupe_abrege'}, inplace=True)

# Sauvegarder le DataFrame final dans un fichier CSV
output_csv = '/Users/eloa/Downloads/xml-4/acteurs_avec_groupe_parlementaire.csv'
final_df.to_csv(output_csv, index=False, encoding='utf-8')

print(f"Fichier CSV généré avec succès : {output_csv}")
