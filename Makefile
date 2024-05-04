PROJECT="Game Interface Flows ML Service"

# make run port=80
run:
	uvicorn api.app:app --host 0.0.0.0 --port $(port) --reload

pretty:
	black .
	isort .

tensorboard:
	tensorboard --logdir=model/tb_logs

docker_build:
	docker image build -t game-interface-flows-ml .

docker_run:
	docker run -p 8001:8001 game-interface-flows-ml

.PHONY: run