LEG16_SEANCES_URL = https://data.assemblee-nationale.fr/static/openData/repository/16/vp/syceronbrut/syseron.xml.zip
LEG16_SEANCES_ZIP = ./data/raw/leg16/syseron.xml.zip
LEG16_SEANCES_DIR = ./data/raw/leg16/seances
LEG16_VOTES_URL = https://data.assemblee-nationale.fr/static/openData/repository/16/loi/scrutins/Scrutins.xml.zip
LEG16_VOTES_ZIP = ./data/raw/leg16/votes.xml.zip
LEG16_VOTES_DIR = ./data/raw/leg16/votes
LEG16_GROUPES_URL = https://data.assemblee-nationale.fr/static/openData/repository/16/amo/acteurs_mandats_organes_divises/AMO50_acteurs_mandats_organes_divises.json.zip
LEG16_GROUPES_ZIP = ./data/raw/leg16/groupes.xml.zip
LEG16_GROUPES_DIR = ./data/raw/leg16/groupes

LEG15_SEANCES_URL = https://data.assemblee-nationale.fr/static/openData/repository/15/vp/syceronbrut/syseron.xml.zip
LEG15_SEANCES_ZIP = ./data/raw/leg15/syseron.xml.zip
LEG15_SEANCES_DIR = ./data/raw/leg15/seances
LEG15_GROUPES_URL = https://data.assemblee-nationale.fr/static/openData/repository/15/amo/deputes_senateurs_ministres_legislature/AMO20_dep_sen_min_tous_mandats_et_organes_XV.xml.zip
LEG15_GROUPES_ZIP = ./data/raw/leg15/AMO20_dep_sen_min_tous_mandats_et_organes_XV.xml.zip
LEG15_GROUPES_DIR = ./data/raw/leg15/groupes

https://data.assemblee-nationale.fr/static/openData/repository/15/amo/deputes_senateurs_ministres_legislature/AMO20_dep_sen_min_tous_mandats_et_organes_XV.xml.zip

push_leg16_files:
	gsutil cp -Z data/leg16-acteur-groupe-famille.csv gs://le-wagon-assnat/
	gsutil cp -Z data/leg16-votes.json gs://le-wagon-assnat/
	gsutil cp -Z data/leg16-seances.csv gs://le-wagon-assnat/
	gsutil cp -Z data/leg16.csv gs://le-wagon-assnat/

download_leg16_files:
	mkdir -p ./data
	gsutil cp gs://le-wagon-assnat/leg16-acteur-groupe-famille.csv data/
	gsutil cp gs://le-wagon-assnat/leg16-votes.json data/
	gsutil cp gs://le-wagon-assnat/leg16-seances.csv data/
	gsutil cp gs://le-wagon-assnat/leg16.csv data/

download_leg16_raw_files:
	mkdir -p ./data/leg16

	mkdir -p $(LEG16_SEANCES_DIR)
	curl -o $(LEG16_SEANCES_ZIP) $(LEG16_SEANCES_URL)
	unzip -o $(LEG16_SEANCES_ZIP) -d $(LEG16_SEANCES_DIR)
	rm -f $(LEG16_SEANCES_ZIP)


	mkdir -p $(LEG16_GROUPES_DIR)
	curl -o $(LEG16_GROUPES_ZIP) $(LEG16_GROUPES_URL)
	unzip -o $(LEG16_GROUPES_ZIP) -d $(LEG16_GROUPES_DIR)
	rm -f $(LEG16_GROUPES_ZIP)


	mkdir -p $(LEG16_VOTES_DIR)
	curl -o $(LEG16_VOTES_ZIP) $(LEG16_VOTES_URL)
	unzip -o $(LEG16_VOTES_ZIP) -d $(LEG16_VOTES_DIR)
	rm -f $(LEG16_VOTES_ZIP)


push_leg15_files:
	gsutil cp -Z data/leg15-acteur-groupe-famille.csv gs://le-wagon-assnat/
	gsutil cp -Z data/leg15-seances.csv gs://le-wagon-assnat/
	gsutil cp -Z data/leg15.csv gs://le-wagon-assnat/

download_leg15_files:
	gsutil cp gs://le-wagon-assnat/leg15-acteur-groupe-famille.csv data/
	gsutil cp gs://le-wagon-assnat/leg15-seances.csv data/
	gsutil cp gs://le-wagon-assnat/leg15.csv data/


download_leg15_raw_files:
	mkdir -p ./data/raw/leg15

	mkdir -p $(LEG15_SEANCES_DIR)
	curl -o $(LEG15_SEANCES_ZIP) $(LEG15_SEANCES_URL)
	unzip -o $(LEG15_SEANCES_ZIP) -d $(LEG15_SEANCES_DIR)
	rm -f $(LEG15_SEANCES_ZIP)

	mkdir -p $(LEG15_GROUPES_DIR)
	curl -o $(LEG15_GROUPES_ZIP) $(LEG15_GROUPES_URL)
	unzip -o $(LEG15_GROUPES_ZIP) -d $(LEG15_GROUPES_DIR)
	rm -f $(LEG15_GROUPES_ZIP)

reset_data_dir:
	@read -p "!!! This will completely delete the ./data directory and download fresh CSV files. Are you sure ? Press ctrl+c to abort or any key to continue" confirmation
	rm -rf ./data
	mkdir -p ./data
	make download_leg16_files
	make download_leg15_files

preprocess_data:
	python -c 'from main import complete_preproc; complete_preproc()'

# DOCKER -- BUILD BACK
docker_build_back_base_local:
	docker build -f Dockerfile-back-base -t assnat-back-base-local .

docker_build_back_base_amd64:
	docker build -f Dockerfile-back-base --platform linux/amd64 -t assnat-back-base-amd64 .

docker_build_back_local:
	docker build -f Dockerfile-back -t assnat-back-local .

docker_build_back_amd64:
	echo Build of image assnat-back-amd64 not automated for now because a lot of env vars to setup. See docker-deploy.sh

# DOCKER -- BUILD FRONT
docker_build_front_base_local:
	docker build -f Dockerfile-front-base -t assnat-front-base-local .

docker_build_front_base_amd64:
	docker build --platform linux/amd64 -f Dockerfile-front-base -t assnat-front-base-amd64 .

docker_build_front_local:
	docker build -f Dockerfile-front -t assnat-front-local .

docker_build_front_amd64:
	docker build --platform linux/amd64 -f Dockerfile-front -t assnat-front-amd64 .

# DOCKER -- RUN
docker_run_back_local:
	docker run -it -p 8000:8080 docker.io/library/assnat-back-local

docker_run_front_local:
	@source .env && docker run -it -e TOKEN_GIF=$$TOKEN_GIF -e API_URL=http://host.docker.internal:8000/predictproba -p 9000:8080 docker.io/library/assnat-front-local


docker_build_all_local:
	docker_build_back_base_local
	docker_build_back_local
	docker_build_front_base_local
	docker_build_front_local

docker_build_all_base_amd64:
	docker_build_back_base_amd64
	docker_build_front_base_amd64
