import torch

checkpoint_path = 'edgeformer/pretrained_models/detection/checkpoint_ema_avg.pt'

layers = {
    'ssd_heads.0.loc_cls_layer.pw_conv.block.conv.weight': 48,
    'ssd_heads.0.loc_cls_layer.pw_conv.block.conv.bias': 48,
    'ssd_heads.1.loc_cls_layer.pw_conv.block.conv.weight': 48,
    'ssd_heads.1.loc_cls_layer.pw_conv.block.conv.bias': 48,
    'ssd_heads.2.loc_cls_layer.pw_conv.block.conv.weight': 48,
    'ssd_heads.2.loc_cls_layer.pw_conv.block.conv.bias': 48,
    'ssd_heads.3.loc_cls_layer.pw_conv.block.conv.weight': 48,
    'ssd_heads.3.loc_cls_layer.pw_conv.block.conv.bias': 48,
    'ssd_heads.4.loc_cls_layer.pw_conv.block.conv.weight': 48,
    'ssd_heads.4.loc_cls_layer.pw_conv.block.conv.bias': 48,
    'ssd_heads.5.loc_cls_layer.block.conv.weight': 32,
    'ssd_heads.5.loc_cls_layer.block.conv.bias': 32,
}

checkpoint = torch.load(checkpoint_path, map_location='cpu')
for key, val in zip(layers.keys(), layers.values()):
    if key.endswith('conv.weight'):
        # print(key)
        # print(checkpoint[key].shape)
        weight_tens = checkpoint[key]
        checkpoint[key] = weight_tens[:val, :, :, :]
    elif key.endswith('conv.bias'):
        # print(key)
        # print(checkpoint[key].shape)
        bias_tens = checkpoint[key]
        checkpoint[key] = bias_tens[:val]

torch.save(checkpoint, 'pretrained/checkpoint_coco_modif.pt')