install_dev:
	pip install -r requirements.txt

build:
	docker build -t footy-ml-starter-kit .

run:
	uvicorn main:app --port 8000 --reload

make run_docker:
	docker run -p 8000:80 footy-ml-starter-kit

train_model:
	python train_model.py



