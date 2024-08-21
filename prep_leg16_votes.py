import xml.etree.ElementTree as ET
import os, json

# Directory containing the XML files
xml_dir = './data/raw/votes/xml'

# Initialize a list to store all the data
all_votes_data = []

namespaces = {'ns': 'http://schemas.assemblee-nationale.fr/referentiel'}

# Loop over each XML file in the directory
for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith('.xml'):
        continue

    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()

    print(xml_file)

    titre = root.find('.//ns:titre', namespaces).text
    print(titre)
    for group in root.findall('.//ns:groupes/ns:groupe', namespaces):
        group_ref = group.find('ns:organeRef', namespaces).text
        print(group_ref)

        pours=[]
        for votant in group.findall('.//ns:decompteNominatif/ns:pours/ns:votant', namespaces):
            acteur_ref = votant.find('ns:acteurRef', namespaces).text
            print(acteur_ref)
            pours.append(acteur_ref)

        contres=[]
        for votant in group.findall('.//ns:decompteNominatif/ns:contres/ns:votant', namespaces):
            acteur_ref = votant.find('ns:acteurRef', namespaces).text
            print(acteur_ref)
            contres.append(acteur_ref)

        all_votes_data.append({
            'titre': titre,
            'groupe': group_ref,
            'pours': pours,
            'contres': contres
        })


with open("data/votes.json", 'w', newline='', encoding='utf-8') as csvfile:
    json.dump(all_votes_data, csvfile, ensure_ascii=False, indent=4)
