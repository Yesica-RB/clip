IMAGE := p4-clip
SCRIPT ?= src/entorno_test.py
K ?= 3

build:
	docker build -t $(IMAGE) .

run:
	docker run -it \
		-v ./:/opt/project \
		-v p4-cache:/root/.cache \
		--rm \
		$(IMAGE) python $(SCRIPT)

shell:
	docker run -it \
		-v ./:/opt/project \
		-v p4-cache:/root/.cache \
		--rm \
		$(IMAGE) /bin/bash

# Atajos para cada parte de la practica
part2:
	$(MAKE) run SCRIPT=src/part2_preclip.py

part3:
	$(MAKE) run SCRIPT=src/part3_clip.py

search:
	$(MAKE) run SCRIPT="src/part4_search.py -k $(K)"
