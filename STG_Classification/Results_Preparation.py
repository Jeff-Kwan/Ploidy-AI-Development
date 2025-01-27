import numpy as np
import matplotlib.pyplot as plt
import json
import os

def F1_score(cm):
    precision = cm[1,1] / (cm[1,1] + cm[0,1] + 1e-12)
    recall = cm[1,1] / (cm[1,1] + cm[1,0] + 1e-12)
    return 2 * precision * recall / (precision + recall + 1e-12)

dir = 'STG_Classification/01-16-Swin'
results = json.load(open(dir+'/01-16_results.json'))

training_losses = results['training_losses']
validation_losses = results['validation_losses']
cms = results['confusion_matrices']

F1_scores = [F1_score(np.array(cm)) for cm in cms]

fig, ax1 = plt.subplots()
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.plot(training_losses, label='Training Loss', color='tab:blue')
# ax1.plot(validation_losses, label='Validation Loss', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('F1 Score', color='tab:red')
ax2.plot(F1_scores, label='Validation F1 Score', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='lower left')
plt.title('Training metrics of swin transformer ploidy classification')
fig.tight_layout()
plt.savefig(os.path.join(dir, f'metrics.png'))

print(f'Best F1 Score: {max(F1_scores)}')
print(f'Last F1 Score: {F1_scores[-1]}')