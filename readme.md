
# EdgeFormer

## Original Repo
https://github.com/hkzhang91/ParC-Net

## Depedencies

```
pip install torch torchvision
pip install opencv-python
pip install pycocotools
pip install pyyaml
```

## Dataset

Download dataset:

3000+ objs: https://drive.google.com/file/d/18cl7RCaEyG4JsIEmbkhczD26D9YG2Afk/view?usp=share_link

6000+ objs: https://drive.google.com/file/d/1D33lX4rWgO2kIgSjthqR75vo4dN6fwf6/view?usp=share_link

1500+ objs: https://drive.google.com/file/d/1FRi7fx1YJ6ciV2b1z8KAqmJhRf2V8ZPQ/view?usp=share_link

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
