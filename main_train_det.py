import sys
import os
import platform

sys.argv.extend(['--common.config-file', 'config/parcnet_det_train.yaml'])

sys.path.append('parcnet')
from parcnet.data import *
from parcnet.main_train import *

if __name__ == "__main__":
    main_worker()
    
    