FROM python:3.8.10-buster

WORKDIR /creed

COPY configs configs
COPY requirements requirements
COPY scripts scripts
COPY src src
COPY train.py train.py
COPY etc/datasets etc/datasets

RUN pip3 install -r requirements/requirements.txt
RUN pip3 install src/
