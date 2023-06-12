import sys
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
sys.argv.extend(['--common.config-file', 'config/edgeformer_det_train.yaml'])

sys.path.append('edgeformer')
from edgeformer.data import *
from edgeformer.main_train import *

if __name__ == "__main__":
    main_worker()
    
    