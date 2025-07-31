.PHONY: exec push run go train release 

CONTAINER = mcoboarr/xgboost
TAG = tag84

.PHONY: build exec push run go train release 

exec:
	docker exec -it $(CONTAINER):$(TAG) /bin/bash
build:
	docker build --platform linux/x86_64 -t $(CONTAINER):$(TAG) .

push:
	docker push $(CONTAINER):$(TAG)

run:
	docker run -it -v "$(PWD):/src" $(CONTAINER):$(TAG) /bin/bash

go:
	make build && make push

clean:
	rm -rf dist/ build/ *.egg-info

