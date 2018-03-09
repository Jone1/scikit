import os
import yaml


def load_yml2env(path):
    stream = open(path)
    docs = yaml.load_all(stream)
    for doc in docs:
        for k, v in doc.items():
            os.environ[k] = str(v)

