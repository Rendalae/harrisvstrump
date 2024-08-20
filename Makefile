SEANCES_CSV = ./data/seances_concatenees_2.csv
GROUPES_CSV = ./data/acteur-groupe.csv
###### 16eme legislature
SEANCES_URL = http://data.assemblee-nationale.fr/static/openData/repository/16/VP/syceronbrut/syseron.xml.zip
GROUPES_URL = https://data.assemblee-nationale.fr/static/openData/repository/16/amo/acteurs_mandats_organes_divises/AMO50_acteurs_mandats_organes_divises.json.zip

SEANCES_ZIP = ./data/syseron.xml.zip
SEANCES_DIR = ./data/raw/seances
GROUPES_ZIP = ./data/groupes.xml.zip
GROUPES_DIR = ./data/raw/groupes

download_csv_files:
	mkdir -p ./data
	gsutil cp gs://le-wagon-assnat/seances_concatenees_2.csv $SEANCES_CSV
	gsutil cp gs://le-wagon-assnat/acteur-groupe.csv $GROUPES_CSV

download_source_files:
	mkdir -p ./data
	mkdir -p $(SEANCES_DIR)
	curl -o $(SEANCES_ZIP) $(SEANCES_URL)
	unzip -o $(SEANCES_ZIP) -d $(SEANCES_DIR)
	rm -f $(SEANCES_ZIP)

	mkdir -p $(GROUPES_DIR)
	curl -o $(GROUPES_ZIP) $(GROUPES_URL)
	unzip -o $(GROUPES_ZIP) -d $(GROUPES_DIR)
	rm -f $(GROUPES_ZIP)
