# Smartloop NLP framework
Natural language processing framework

# Train a bot

Use the `sample.json`  file in the `\data` folder, you will pass the name of bot as an argument in the next step. 

Below is as training JSON sample  containing the pattern and name of the intent that wil be resolved for a user input.

```json
{
    "examples": {
        "intents": [
            {
                "text": "about",
                "intent": "about"
            },
            {
                "text": "company",
                "intent": "about"
            },
            {
                "text": "what is smartloop",
                "intent": "about"
            },
            {
                "text": "start",
                "intent": "start"
            },
            {
                "text": "menu",
                "intent": "start"
            },
            {
                "text": "hi",
                "intent": "start"
            }
        ]
    },
    "lang": "en"
}
```

From the command line type the following to train the bot:

```
python main.py train -i sample

```

Testing the bot

To test the type the following command:

```
python main.py parse -i sample -t "I need a chabot"
```

This should return the intent name followed by the confidence level

```
{
    "topIntent": {
        "intent": "i-need-chatbot",
        "confidence": 0.9999436140060425
    },
    "intents": [
        {
            "intent": "i-need-chatbot",
            "confidence": 0.9999436140060425
        },
        {
            "intent": "chatter-good-afternoon",
            "confidence": 4.835660001845099e-05
        },
        {
            "intent": "bizbot-no-way",
            "confidence": 3.6056665067008e-06
        },
        {
            "intent": "about-chatbot",
            "confidence": 1.9573460576793877e-06
        },
        {
            "intent": "contact",
            "confidence": 1.095663265004987e-06
        }
    ]
}
```

## Tunning your model (Advanced)

It is possible to override the default training parameters to create a model that fits your need, override `config.yaml` to tune your model:

```yaml
# number of epochs
epochs: 100

# Use tensorboard callback
logs: True

# classifier parameters
embedded_intent_classifier:
    # base neurons, this will be increased based on the intent size
    neurons: 16
    # length of input len("hello how are you") = 4
    input_length: 100
    learning_rate: 1e-2
    flatten: False
    hidden_layers: 2
    # drop rate to avoid overfitting
    drop_rate: 0.2
    # early stop training in case of not improving
    early_stopping: True
```

This can vary based on model size, can be tuned using the grid search capabablites to find the optimal settings. 

Here is a list of basic parameters and their meaning:

* epochs - This is the number of iterations where 1 epoch = 1 complete neural net cycle
* learning_rate - How fast or slow, the model is learning through iterations
* drop_rate - Adjust to prevent overfitting of the data to fine tune your model


## Configuration

Install stop words dictionary using following command

```
python -m nltk.downloader stopwords   
```

## Debugging

Set `logs:True` in config.yaml to enable debugging using `tensorboard`. Once you have trained the bot. Type the following command to start tensorboard:

```commandline
tensorboard serve --logdir logs/nlp_data/<bot_id>/<model_id>
```


## Requirements

* Tensorflow (>=2.12.0)

## License
Licensed under the Apache License, Version 2.0. 

Copyright 2021-2023 Smartloop Inc.
