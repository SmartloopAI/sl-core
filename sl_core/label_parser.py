from sklearn.preprocessing import LabelEncoder

# max number of intents to return
INTENT_THRESHOLD = 5


class LabelParser:
    """
        Label parser
    """

    def __init__(self, classes):
        self.classes = classes

    def parse(self, pred):
        if len(pred) > 0:
            intents = []
            args = pred[0]

            for i in range(len(args)):
                intents.append({
                    'name': self.classes[i],
                    'confidence': float(args[i])
                })

            idx = args.argmax()

            return {
                'topIntent': {
                    'name': self.classes[idx],
                    'confidence': float(args[idx])
                },
                'intents': sorted(intents, key=lambda x: x['confidence'], reverse=True)[:INTENT_THRESHOLD]
            }
        return {
            'fallback': float(0.0)
        }
