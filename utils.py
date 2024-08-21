import os

# Liste des fichiers JSON dans un folder
# extension : .json, .xml, .csv...
def list_files(folder,extension : str):
    json_files=[]
    for file_name in os.listdir(folder):
            if not file_name.endswith(extension):
                continue
            json_path = os.path.join(folder, file_name)
            json_files.append(json_path)
    return json_files

# Ajouter une valeur à une clé d'un dictionnaire de liste
def append_to_dict_key(dict, key, val):
    if key not in dict:
        dict[key] = []
    dict[key].append(val)
