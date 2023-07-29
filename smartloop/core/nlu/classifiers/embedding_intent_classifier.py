import json
import os
import logging
import joblib
import numpy as np

from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping

from smartloop.core import Config, DefaultConfig

from smartloop.core.classifier import Classifier
from smartloop.core.label_parser import LabelParser
from smartloop.core.tokenizer import SentenceTokenizer
from smartloop.core.nlu import OOV_TOKEN
from smartloop.core.nlu.callbacks import TrainStatusReport
from smartloop.core.model_loader import ModelLoader

from smartloop.core.sanitizer import Sanitizer

logger = logging.getLogger(__name__)


class EmbeddingIntentClassifier(Classifier):
    """
        Intent classifier using LSTM model
    """

    cfg: Config

    def __init__(self, project_dir=None, lang='en', classes=None):
        self.project_dir = project_dir
        self.classes = classes
        self.lang = lang
        self.cfg = DefaultConfig()

    def set_config(self, cfg: Config):
        self.cfg = cfg

    def fit(self, X, y):
        cfg = self.cfg.embedded_intent_classifier

        vocab_size = len(X)

        neurons = int(cfg.get("neurons", 32))
        drop_rate = float(cfg.get("drop_rate", 0.2))
        flatten = bool(cfg.get("flatten", True))
        hidden_layers = int(cfg.get("hidden_layers", 1))
        learning_rate = float(cfg.get("learning_rate", 1e-2))

        label_encoder = LabelEncoder()
        label_encoder.fit(y)

        tokenizer = Tokenizer(lower=True, num_words=vocab_size, oov_token=OOV_TOKEN)

        sent_tokenizer = SentenceTokenizer(tokenizer)

        sent_tokenizer.max_input_length = np.array(X).dtype.itemsize

        sent_tokenizer.fit(X)

        sequences = sent_tokenizer.transform(X)

        output_dim = len(label_encoder.classes_)

        # build the model
        model = self.__build_model(
            input_dim=len(tokenizer.word_counts),
            output_dim=output_dim,
            input_length=sequences.shape[1],
            neurons=neurons,
            drop_rate=drop_rate,
            flatten=flatten,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
        )

        logging.info(model.summary())

        self.__fit(model, sequences, label_encoder.transform(y))

        ml = ModelLoader(self.project_dir)

        ml.save(model)

        ml.append_metadata({
            'labels': json.dumps({
                'values': label_encoder.classes_.tolist()
            }),
            'max_input_length': sent_tokenizer.max_input_length,
            'lang': self.lang,
            'version': '1.0',
            'tokens': json.dumps(tokenizer.to_json())
        })

    def __fit(self, model, features, intent):
        # fit the model
        log_dir = "logs/{}".format('_'.join(os.path.split(self.project_dir)[:1]))

        callbacks = [TrainStatusReport(report_every=10)]

        if self.cfg.logs:
            callbacks.append(TensorBoard(log_dir=log_dir))

        early_stopping = self.cfg.embedded_intent_classifier.get('early_stopping', False)

        if early_stopping:
            callbacks.append(EarlyStopping(monitor='accuracy', mode='max', patience=10))

        model.fit(
            features,
            intent,
            validation_split=0.2,
            epochs=self.cfg.epochs,
            callbacks=[
                callbacks
            ],
            verbose=0
        )

    def transform(self, X):
        try:
            bm = ModelLoader(self.project_dir)
            model, metadata = bm.load(attrs=['labels', 'max_input_length', 'lang', 'tokens', 'version'])

            # check if meta_data exists
            if metadata is not None:
                max_input_length = metadata.get('max_input_length', 100)
                lang = metadata.get('lang')
                version = metadata.get('version')

                y = metadata['labels'].get('values', [])

                if version is None:
                    with open("{}/tokenizer.json".format(self.project_dir), "r+") as f:
                        tokenizer = tokenizer_from_json(f.read())
                    with open(os.path.join(self.project_dir, 'meta.json'), "r") as f:
                        meta = json.loads(f.read())
                        sanitizer = Sanitizer(lang=meta.get('lang', 'en'))
                else:
                    sanitizer = Sanitizer(lang=lang)
                    tokenizer = tokenizer_from_json(metadata.get('tokens'))
            else:
                max_input_length = self.cfg.embedded_intent_classifier.get("input_length", 100)
                y = joblib.load("{}/classes.pkl".format(self.project_dir))

            X_test = [sanitizer.transform(sent) for sent in X]

            # sentence tokenizer
            sent_tokenizer = SentenceTokenizer(tokenizer)
            sent_tokenizer.max_input_length = max_input_length

            # vectorize
            vec = sent_tokenizer.transform(X_test)
            # predict for input
            pred = model.predict(vec)
            # parse the label
            parser = LabelParser(y)
            # return the result
            return parser.parse(pred)
        except ValueError:
            logger.fatal("failed load model from disk")

    def __build_model(self, input_dim, output_dim, input_length, neurons, drop_rate,
                      flatten,
                      hidden_layers,
                      learning_rate):
        model = keras.Sequential()

        embedding_layer = keras.layers.Embedding(input_dim=input_dim + 1, output_dim=output_dim,
                                                 input_length=input_length)

        model.add(embedding_layer)

        model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(neurons)
            )
        )

        model.add(keras.layers.Dropout(drop_rate))

        # flatten
        if flatten:
            model.add(keras.layers.Flatten())

        # hidden layers
        for i in range(hidden_layers):
            # hidden layer
            model.add(keras.layers.Dense(
                model.output_shape[1],
                kernel_regularizer=keras.regularizers.L2(l2=0.002)
            ))

        model.add(keras.layers.Dense(output_dim, activation="softmax"))

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=100,
            decay_rate=0.9)

        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(),
            optimizer=optimizer,
            metrics=['accuracy'],
        )

        return model
