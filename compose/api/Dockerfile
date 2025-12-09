FROM python:3.12-slim

RUN apt-get update
#   dependencies building Python packages \
#  && apt-get install -y build-essential \
#   psycopg2 dependencies

WORKDIR /app

#COPY environment.yml .

#RUN pip env create -f environment.yml

COPY requirements_demo.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY src ./

FROM python:3.12-slim

RUN apt-get update
#   dependencies building Python packages \
#  && apt-get install -y build-essential \
#   psycopg2 dependencies

WORKDIR /app

#COPY environment.yml .

#RUN pip env create -f environment.yml

COPY requirements_demo.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY src ./

FROM python:3.12-slim

RUN apt-get update
#   dependencies building Python packages \
#  && apt-get install -y build-essential \
#   psycopg2 dependencies

WORKDIR /app

#COPY environment.yml .

#RUN pip env create -f environment.yml

COPY requirements_demo.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY src ./src

CMD ["streamlit", "run", "src/main.py"]

