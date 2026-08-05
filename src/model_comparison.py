import pandas as pd
import matplotlib.pyplot as plt

# Load the auto-saved comparison file (built by LR, RF, XGBoost scripts)
comparison_df = pd.read_csv("models/model_comparison.csv")

print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)
print(comparison_df.to_string(index=False))

# Save a clean markdown table for dissertation use
comparison_df.to_markdown("models/model_comparison_table.md", index=False)
print("\n[INFO] Comparison table saved as markdown for dissertation.")

# Comparison bar chart (all 4 metrics, all 3 models)
metrics = ['auprc', 'auc_roc', 'precision_aki', 'recall_aki']
metric_labels = ['AUPRC', 'AUC-ROC', 'Precision (AKI)', 'Recall (AKI)']

fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(metrics))
width = 0.25
colors = ['#4C72B0', '#55A868', '#8172B2']

for i, model_name in enumerate(comparison_df['model']):
    values = comparison_df[comparison_df['model'] == model_name][metrics].values[0]
    ax.bar([xi + i*width for xi in x], values, width, label=model_name, color=colors[i])

ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1)
ax.set_ylabel('Score')
ax.set_title('Model Comparison: LR vs Random Forest vs XGBoost')
ax.legend()
plt.tight_layout()
plt.savefig("models/model_comparison_chart.png", dpi=150)
plt.show()