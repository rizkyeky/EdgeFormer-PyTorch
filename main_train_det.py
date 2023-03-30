import sys
import os
import platform

sys.argv.extend(['--common.config-file', 'config/parcnet_det_train.yaml'])

# system = platform.system()
# path = os.getcwd()
# if (system == 'Linux'):
#     if (path.endswith('content')): # Colab
#         sys.argv.extend(['--common.config-file', 'config/parcnet_det_train.yaml'])
#     elif (path.endswith('working')): # Kaggle
#         sys.argv.extend(['--common.config-file', 'config/parcnet_det_train.yaml'])
#     elif (path.endswith('parcnet')): # LabAI
#         sys.argv.extend(['--common.config-file', 'config/parcnet_det_train.yaml'])
#         sys.argv.extend(['--dataset.root_train', 'dataset'])
#         sys.argv.extend(['--dataset.root_val', 'dataset'])
#         sys.argv.extend(['--model.classification.pretrained', 'checkpoints/checkpoint_last_93.pt'])
#         sys.argv.extend(['--model.detection.pretrained', 'checkpoints/checkpoint_last_run19.pt'])
# elif (system == 'Darwin' and path.endswith('_main')): # MacOS Eky
#     sys.argv.extend(['--common.config-file', 'config/parcnet_det_train.yaml'])
#     sys.argv.extend(['--model.classification.pretrained', 'parcnet/pretrained_models/classification/checkpoint_last_93.pt'])
#     sys.argv.extend(['--model.detection.pretrained', 'parcnet/pretrained_models/detection/unbalance/checkpoint_last_run17.pt'])

sys.path.append('parcnet')
from parcnet.data import *
from parcnet.main_train import *

if __name__ == "__main__":
    main_worker()
    
    