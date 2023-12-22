import json
import logging
import argparse
import time
import timeit
import os

from smartloop.core import FileConfig
from smartloop.core import Project
from smartloop.core.text_classifier import TextClassifier

logging.basicConfig()
logging.root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # tensorflow random seed
    # start time
    start_time = timeit.default_timer()
    # process args
    parser = argparse.ArgumentParser(description="Smartloop NLP")
    parser.add_argument(
        "command",
        metavar="<command>",
        choices=["train", "parse"],
        help="argument: train|parse",
    )
    parser.add_argument("-i", "--id", dest="id")
    parser.add_argument("-l", "--lang", dest="lang", default='en')
    parser.add_argument("-b", "--best", dest="use_best", action='store_true')
    parser.add_argument("-t", "--text", dest="text")

    args = parser.parse_args()

    # TODO: parameterize
    data_dir = os.getenv('DATA_DIR', 'nlp_data')

    # load config from disk
    cfg = FileConfig()

    if args.command == "train":
        if args.id is not None:
            proj = Project(data_dir=data_dir, project_id=args.id, model_id=str(int(time.time())))
            cls = TextClassifier(proj=proj, cfg=cfg)
            with open('data/%s.json' % args.id, 'r+') as f:
                data = json.loads(f.read())
            cls.fit(data)
    elif args.command == "parse":
        if args.id is not None and args.text is not None:
            # taking the latest model
            proj = Project(data_dir=data_dir, project_id=args.id, model_id="1")
            cls = TextClassifier(proj=proj, cfg=cfg)
            result = cls.transform([args.text])
            print(json.dumps(result, indent=4))

    end_time = timeit.default_timer() - start_time

    logger.info('Completed in %s seconds' % end_time)
