PROJECT="Game Interface Flows ML Service"

run:
	uvicorn app.app:app --reload

.PHONY: run