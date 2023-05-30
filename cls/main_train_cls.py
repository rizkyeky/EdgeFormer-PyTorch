import sys

sys.argv.extend(['--common.config-file', 'config/parcnet_cls_train.yaml'])

sys.path.append('parcnet')
from parcnet.main_train import *

if __name__ == "__main__":
    main_worker()
    
    