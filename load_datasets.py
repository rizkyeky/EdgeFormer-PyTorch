from datasets import load_dataset

dataset = load_dataset("detection-datasets/coco")
print(dataset[0])