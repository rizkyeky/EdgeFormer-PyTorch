import matplotlib.pyplot as plt
import json
import sys
import numpy as np

with open('results/run10000/{}'.format(sys.argv[1]), 'r') as f:
    history = json.load(f)


train_loss = history['train_avg_loss']
val_loss = history['val_avg_loss']

print(train_loss[0:10], train_loss[-10:])

plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(np.arange(0,len(history['train_avg_loss'])+1,500))
plt.legend()

plt.show()

