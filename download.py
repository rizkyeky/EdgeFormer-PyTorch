import gdown

train_url = 'https://drive.google.com/uc?id=1Sm6hEJVb6G0N0uLxw1vxkQnxqy1WSKsH'
val_url = 'https://drive.google.com/uc?id=1VXt0_BNw4MQQOzvH5jO9pcbcJkMHJ1F3'
ann_url = 'https://drive.google.com/uc?id=1gaXbrjUEV6UlKnjqQmw8tWYN7D4t8m93'
checkpoint_cls = 'https://drive.google.com/uc?id=1tXAZto-WKwhkkNv6x43KI6mcbfAdyRBI'
# checkpoint_det = 'https://drive.google.com/uc?id=1-BCVJq2-rvOIFjCt5HUrK7lTnnXhxciu' # 19
checkpoint_det = 'https://drive.google.com/uc?id=1-IIBlEIdhQmHuO-jlAoNNLWBqY8FTHqO' # 20

train_output = 'dataset/images/un_train.zip'
val_output = 'dataset/images/un_val.zip'
ann_output = 'dataset/annotations/annotations.zip'
chdet_output = 'checkpoints/checkpoint_last_run19.pt'
chcls_output = 'checkpoints/checkpoint_last_93.pt'

# gdown.download(train_url, train_output)
# gdown.download(val_url, val_output)
# gdown.download(ann_url, ann_output)
# gdown.download(checkpoint_cls, chcls_output)
gdown.download(checkpoint_det, chdet_output)