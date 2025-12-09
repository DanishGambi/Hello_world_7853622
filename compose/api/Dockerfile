FROM continuumio/miniconda

RUN apt-get update \
  # dependencies for building Python packages \
  && apt-get install -y build-essential \
  # psycopg2 dependencies

WORKDIR usr/app

COPY ./environment.yml usr/app/

CMD conda create -f environment.yml

COPY /src /usr/app/

