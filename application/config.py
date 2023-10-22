import os

os.environ["GRPC_VERBOSITY"] = "info"
os.environ["GRPC_TRACE"] = "http_keepalive"


SERVER_ADDRESS = os.environ['SERVER_ADDRESS']
HOST = os.environ['HOST']
PORT = os.environ['PORT']
FEDERATED_PORT = os.environ['FEDERATED_PORT']
REPOSITORY_ADDRESS = os.environ['REPOSITORY_ADDRESS']
ORCHESTRATOR_ADDRESS = os.environ['ORCHESTRATOR_ADDRESS']
JSON_FILE = os.environ['JSON_FILE']
HM_SECRET_FILE = os.environ['HM_SECRET_FILE']
HM_PUBLIC_FILE = os.environ['HM_PUBLIC_FILE']
