FROM python:3.8.3-slim-buster
WORKDIR /code
ENV GRPC_TRACE=all
ENV GRPC_VERBOSITY=DEBUG
COPY requirements.txt /code/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . /code/application
ENV PYTHONPATH "${PYTHONPATH}:/code/application"
EXPOSE 8000
EXPOSE 8080
ENTRYPOINT ["python3", "./application/main.py"]