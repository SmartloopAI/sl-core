import json

import h5py
import numpy
import numpy as np

from tensorflow import keras


class ModelLoader(object):

    def __init__(self, project_dir, override=True):
        self.filepath = "{}/{}.h5".format(project_dir, "model")
        self.override = override

    # saves the model with metadata
    def save(self, model):
        keras.models.save_model(model, self.filepath, self.override)

    def append_metadata(self, meta_data=None):
        if meta_data is None:
            meta_data = dict()

        f = h5py.File(self.filepath, mode='a')

        for k, v in meta_data.items():
            f.attrs[k] = v

        f.close()

    def load(self, attrs=None):
        if attrs is None:
            attrs = []
        model = keras.models.load_model(self.filepath)
        # load as hdfs
        f = h5py.File(self.filepath, mode='r')

        meta_data = None

        for i in range(len(attrs)):
            key = attrs[i]
            if key in f.attrs:
                if meta_data is None:
                    meta_data = dict()

                val = f.attrs.get(key)

                if type(val) == numpy.int64:
                    meta_data[key] = val
                else:
                    try:
                        meta_data[key] = json.loads(val)
                    except ValueError:
                        # failed
                        meta_data[key] = val

        return model, meta_data
