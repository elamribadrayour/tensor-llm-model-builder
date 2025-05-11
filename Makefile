build:
	echo "Building docker image"
	docker compose build

tag:
	echo "Tagging docker image"
	docker tag tensorrt-llm-model-builder-tensorrt-llm-model-builder:latest ghcr.io/elamribadrayour/tensorrt-llm-model-builder:25.03-trtllm-python-py3

push:
	echo "Pushing docker image"
	docker push ghcr.io/elamribadrayour/tensorrt-llm-model-builder:25.03-trtllm-python-py3

deploy:
	echo "Deploying docker image"
	make build
	make tag
	make push
