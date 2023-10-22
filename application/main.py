import os
import pickle
import signal
import uvicorn
from fastapi import FastAPI
from flwr.common import logger
import logging

from config import HOST, JSON_FILE, PORT
from datamodels.models import StatusEnum, TCTrainingConfiguration, Status
from multiprocessing import Manager, Process, set_start_method
from src import training_server

app = FastAPI()
manager = Manager()
if os.path.isfile(os.path.join("..", JSON_FILE)):
    with open(os.path.join("..", JSON_FILE), 'rb') as handle:
        prev = pickle.load(handle)
        app.jobs = manager.dict(prev)
else:
    app.jobs = manager.dict()


@app.post("/job/config/{training_id}/")
async def receive_conf(training_id, data: TCTrainingConfiguration):
    '''
    Receive training configuration
    '''
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    try:
        p = Process(name=str(training_id), target=training_server.start_flower_server, args=(
            training_id, data, app.jobs))
        p.start()
        app.jobs[training_id] = Status(
            status=StatusEnum.WAITING, round=-1, id=p.pid)
    except Exception as e:
        print("An exception occurred ::", e)
        return 500
    return 200


@app.get("/job/status/{training_id}")
async def retrieve_status(training_id):
    '''
    Receive job status updates
    '''
    if training_id in app.jobs:
        return app.jobs[training_id]
    else:
        return Status()


@app.post("/job/stop")
async def stop_job():
    '''
    For now, let's make it stop all jobs that are not stopped. 
    '''
    try:
        inactive_statuses = [StatusEnum.FINISHED,
                             StatusEnum.INACTIVE, StatusEnum.INTERRUPTED]
        for training_id in app.jobs:
            if app.jobs[training_id] not in inactive_statuses:
                os.kill(app.jobs[training_id].id, signal.SIGKILL)
                app.jobs[training_id] = StatusEnum.INTERRUPTED
    except Exception as e:
        print("An exception occurred ::", e)
        return 500
    return 200


if __name__ == "__main__":
    logger.logger.setLevel(logging.INFO)
    uvicorn.run(app, host=HOST, port=int(PORT))
