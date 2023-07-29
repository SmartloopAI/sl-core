import json
import os

import numpy as np

from smartloop.core import Config
from smartloop.core import Project

from smartloop.core.nlu.classifiers import EmbeddingIntentClassifier
from smartloop.core.sanitizer import Sanitizer


class TextClassifier(object):
    """
        Trains and parses input data to resolve an intent for a given project_id
    """

    def __init__(self, proj: Project, cfg: Config = Config()):
        """
        :type project: Project
        :type cfg: Config
        """

        self.proj = proj
        self.cfg = cfg

    def __get_classifier(self, project_dir, lang='en'):
        cls = EmbeddingIntentClassifier(
            project_dir=project_dir,
            lang=lang
        )

        cls.set_config(self.cfg)

        return cls

    def fit(self, data):
        project_dir = self.proj.get_project_dir(training=True)

        examples = data['examples']

        intents = np.asarray(examples['intents'])

        lang = data['lang']

        sanitizer = Sanitizer(data['lang'])

        cls = self.__get_classifier(project_dir, lang)

        extract_text = np.vectorize(lambda x: sanitizer.transform(x['text']))

        extract_intents = np.vectorize(lambda x: x['intent'])

        X = extract_text(intents)
        y = extract_intents(intents)

        mask = np.ones(len(X), dtype=bool)
        mask[np.unique(X, return_index=True)[1]] = False

        X = X[~mask]
        y = y[~mask]

        cls.fit(X, y)

    def transform(self, X):
        project_dir = self.proj.get_project_dir()
        cls = self.__get_classifier(project_dir)
        return cls.transform(X)
