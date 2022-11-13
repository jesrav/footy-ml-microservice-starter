install_dev:
	pip install -r requirements.txt

run:
	uvicorn main:app --port 8000 --reload

