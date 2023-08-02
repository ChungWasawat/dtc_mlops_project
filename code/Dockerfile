FROM python:3.9-slim

RUN pip install -U pip
RUN pip install jinja2
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

ENV RUN_ID=33f0efafe8de40f8aa6b4acd0a381b26

COPY [ "model_deployment.py", "others/final_preprocessor.b",  "./" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "model_deployment:app" ]