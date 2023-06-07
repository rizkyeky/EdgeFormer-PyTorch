import matplotlib.pyplot as plt
import json
import sys
import numpy as np

file = 'results/run10000/history.json'
if len(sys.argv) > 1:
    file = 'results/{}'.format(sys.argv[1])

with open(file, 'r') as f:
    history = json.load(f)

train_loss = history['train_avg_loss']
val_loss = history['val_avg_loss']

chunk_size = 1000
train_means = np.convolve(train_loss, np.ones(chunk_size)/chunk_size, mode='same')
val_means = np.convolve(val_loss, np.ones(chunk_size)/chunk_size, mode='same')

plt.plot(train_loss, label='Training Loss')
# plt.plot(train_means, label='Training Loss Mean')
plt.plot(val_loss, label='Validation Loss')
# plt.plot(val_means, label='Training Loss Mean')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.xticks(np.arange(0,len(history['train_avg_loss'])+1,500))
plt.legend()

plt.show()

