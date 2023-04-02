import matplotlib.pyplot as plt
import json

with open('results/run10000/history.json', 'r') as f:
    history = json.load(f)

plt.plot(history['train_avg_loss'], label='Training Loss')
plt.plot(history['val_avg_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
