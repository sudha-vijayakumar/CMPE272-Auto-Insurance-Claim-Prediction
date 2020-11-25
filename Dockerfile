FROM python:3.5.2-slim

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app
COPY encoded.csv /usr/src/app
COPY trained_model.pkl /usr/src/app

RUN pip install -r requirements.txt

COPY . /usr/src/app

EXPOSE 5000

CMD ["python", "./main.py" ]

