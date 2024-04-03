PROJECT="Game Interface Flows ML Service"

# make run port=80
run:
	uvicorn api.app:app --host 0.0.0.0 --port $(port) --reload

pretty:
	black .
	isort .

.PHONY: run