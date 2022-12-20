FROM python:3.10.8-buster

WORKDIR /creed

COPY configs configs
COPY requirements requirements
COPY scripts scripts
COPY src src

RUN pip3 install -r requirements/requirements.txt
RUN pip3 install src/
