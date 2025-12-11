#FROM python:3.12-slim
#
#RUN apt-get update
##   dependencies building Python packages \
##  && apt-get install -y build-essential \
##   psycopg2 dependencies
#
#WORKDIR /app
#
##COPY environment.yml .
#
##RUN pip env create -f environment.yml
#
#COPY requirements_demo.txt requirements.txt
#
#RUN pip install --upgrade pip
#RUN pip install -r requirements.txt
#
#COPY src ./src
#
#CMD ["streamlit", "run", "src/main.py"]

############################

FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.yml .

RUN conda env update -f environment.yml && \
    conda clean --all --yes

ENV PATH /opt/conda/envs/hakaton/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre && \
    rm -rf /var/lib/apt/lists/*

COPY src ./src

CMD ["streamlit", "run", "src/main.py"]




