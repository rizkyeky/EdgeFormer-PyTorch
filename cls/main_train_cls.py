import sys

sys.argv.extend(['--common.config-file', 'config/edgeformer_cls_train.yaml'])

sys.path.append('edgeformer')
from edgeformer.main_train import *

if __name__ == "__main__":
    main_worker()
    
    