import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ["Model Accuracy", "Rule Accuracy", "Rule Fidelity"]
auto_values = [89.49, 90.67, 99.64]
or_values = [87.96, 76.29, 98.83]

# Bar chart setup
x = np.arange(len(labels))

x = np.array([1.4, 2, 2.6])

# Updated colors and bar width
colors = ['#ff7f0e', '#1f77b4']
width = 0.2

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, auto_values, width, label='Auto', color=colors[0])
rects2 = ax.bar(x + width/2, or_values, width, label='OR', color=colors[1])

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Metrics')
ax.set_ylabel('(%)')
# ax.set_title('Comparison between using the automatically decided connector and OR on CUB dataset', wrap=True)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Autolabel function for displaying the value on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 1),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.savefig(f"./ablation.jpg")
plt.savefig(f"./ablation.pdf")

plt.show()
