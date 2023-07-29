import time
import json
import logging

from smartloop_core import FileConfig
from smartloop_core import Project
from smartloop_core.text_classifier import TextClassifier

project_id = 'sample'
data_dir = '../nlp_data'

logging.basicConfig()
logging.root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

def test_sample_model():
    cfg = FileConfig()

    mode_id = int(time.time())

    proj = Project(data_dir=data_dir, project_id=project_id, model_id=str(mode_id))
    cls = TextClassifier(proj=proj, cfg=cfg)

    with open('data/{}.json'.format(project_id), 'r') as f:
        data = json.loads(f.read())

    cls.fit(data)

    res = cls.transform(['hi'])

    assert res is not None
    assert res['topIntent']['intent'] == 'start'
