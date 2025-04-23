import logging
from lib import evaluation

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Evaluate model ensemble
paths = ['runs/coco_t2i/results_coco.npy',
         'runs/coco_i2t/results_coco.npy']

evaluation.eval_ensemble(results_paths=paths, fold5=True)
# evaluation.eval_ensemble(results_paths=paths, fold5=False)
