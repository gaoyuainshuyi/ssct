import logging
from lib import evaluation

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Evaluate model ensemble
paths = ['runs/f30k_t2i/results_f30k.npy',
         'runs/f30k_i2t/results_f30k.npy']

# evaluation.eval_ensemble(results_paths=paths, fold5=True)
# evaluation.eval_ensemble(results_paths=paths, fold5=False)
evaluation.eval_ensemble(results_paths=paths)
