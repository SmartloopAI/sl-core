import tensorflow as tf

from .config import Config
from .default_config import DefaultConfig
from .file_config import FileConfig
from .project import Project
from .model_loader import ModelLoader

LANG_MAPS = {
    'en': 'english',
    'es': 'spanish',
    'de': 'german'
}

tf.random.set_seed(hash('smartloop'))
