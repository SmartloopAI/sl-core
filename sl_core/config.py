class Config(object):
    def __init__(self, epochs=10, logs=False, embedded_intent_classifier=None, early_stopping=False):
        if embedded_intent_classifier is None:
            embedded_intent_classifier = dict()
        self.epochs = epochs
        self.logs = logs
        self.early_stopping = early_stopping
        self.embedded_intent_classifier = embedded_intent_classifier
