import yaml
from .config import Config


class FileConfig(Config):
    def __init__(self, file_name="config.yaml"):
        with open(file_name, "r+") as f:
            cfg = yaml.safe_load(f)
            super().__init__(**cfg)
