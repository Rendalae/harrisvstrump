GROUPES_CSV = ./data/acteur-groupe.csv
SEANCES_CSV = ./data/seances_concatenees_2.csv
SEANCES_CSV2 = ./data/seances_concatenees_2_withgroupepol.csv
VOTES_JSON = ./data/votes.json
###### 16eme legislature
SEANCES_URL = https://data.assemblee-nationale.fr/static/openData/repository/16/vp/syceronbrut/syseron.xml.zip
GROUPES_URL = https://data.assemblee-nationale.fr/static/openData/repository/16/amo/acteurs_mandats_organes_divises/AMO50_acteurs_mandats_organes_divises.json.zip
VOTES_URL = https://data.assemblee-nationale.fr/static/openData/repository/16/loi/scrutins/Scrutins.xml.zip

SEANCES_ZIP = ./data/syseron.xml.zip
SEANCES_DIR = ./data/raw/seances
GROUPES_ZIP = ./data/groupes.xml.zip
GROUPES_DIR = ./data/raw/groupes
VOTES_ZIP = ./data/votes.xml.zip
VOTES_DIR = ./data/raw/votes

push_csv_files:
	mkdir -p ./data
	gsutil cp data/leg16-acteur-groupe-famille.csv gs://le-wagon-assnat/
	gsutil cp data/leg16-votes.json gs://le-wagon-assnat/
	gsutil cp data/leg16-seances.csv gs://le-wagon-assnat/
	gsutil cp data/leg16.csv gs://le-wagon-assnat/

download_csv_files:
	mkdir -p ./data
	gsutil cp gs://le-wagon-assnat/leg16-acteur-groupe-famille.csv data/
	gsutil cp gs://le-wagon-assnat/leg16-votes.json data/
	gsutil cp gs://le-wagon-assnat/leg16-seances.csv data/
	gsutil cp gs://le-wagon-assnat/leg16.csv data/

download_raw_seances:
	mkdir -p ./data
	mkdir -p $(SEANCES_DIR)
	curl -o $(SEANCES_ZIP) $(SEANCES_URL)
	unzip -o $(SEANCES_ZIP) -d $(SEANCES_DIR)
	rm -f $(SEANCES_ZIP)

download_raw_groupes:
	mkdir -p ./data
	mkdir -p $(GROUPES_DIR)
	curl -o $(GROUPES_ZIP) $(GROUPES_URL)
	unzip -o $(GROUPES_ZIP) -d $(GROUPES_DIR)
	rm -f $(GROUPES_ZIP)

download_raw_votes:
	mkdir -p ./data
	mkdir -p $(VOTES_DIR)
	curl -o $(VOTES_ZIP) $(VOTES_URL)
	unzip -o $(VOTES_ZIP) -d $(VOTES_DIR)
	rm -f $(VOTES_ZIP)
