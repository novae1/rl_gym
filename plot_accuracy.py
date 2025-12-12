import matplotlib.pyplot as plt

# Data from eval results
base_accuracy = 52.76724791508719
grpo_accuracy = 55.26914329037149

# Create bar plot
fig, ax = plt.subplots(figsize=(8, 6))
models = ['Base', 'GRPO']
accuracies = [base_accuracy, grpo_accuracy]
colors = ['#3498db', '#2ecc71']

bars = ax.bar(models, accuracies, color=colors, width=0.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Customize plot
ax.set_ylabel('Performance (%)', fontsize=12)
ax.set_title('Comparação de Performance no GSM8K', fontsize=14, fontweight='bold')
ax.set_ylim(0, 70)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('gsm8k_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print('Plot saved as gsm8k_accuracy_comparison.png')
