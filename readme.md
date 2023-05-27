
# EdgeFormer

## Depedencies

```
pip install torch torchvision
pip install opencv-python
pip install pycocotools
pip install pyyaml
```

## Dataset

Create dataset folder and follow the sturcture folder like this

```        
├── dataset
│   ├── images
│   │   ├── train/*.jpg
|   |   ├── val/*.jpg
│   └── annotations
│       ├── train.json
│       └── val.json
│   
├── main_train_det.py
```

## Train
```
python main_train_det.py
```
