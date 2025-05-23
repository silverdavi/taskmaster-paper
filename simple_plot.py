import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('analysis_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Create simple plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Model performance
models = ['Keras NN', 'Random Forest'] 
test_r2 = [results['results']['keras']['test']['r2'], results['results']['random_forest']['test']['r2']]

ax1.bar(models, test_r2, color=['orange', 'green'], alpha=0.8)
ax1.set_ylabel('Test R² Score')
ax1.set_title('Model Performance (Episode-Level)')
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Baseline')
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend()

# Feature importance
feature_names = list(results['feature_importance'].keys())
importance_values = list(results['feature_importance'].values())

ax2.barh(range(len(feature_names)), importance_values, color='skyblue', alpha=0.8)
ax2.set_yticks(range(len(feature_names)))
ax2.set_yticklabels([name.replace('avg_', '').title() for name in feature_names])
ax2.set_xlabel('Importance Score')
ax2.set_title('Feature Importance (Random Forest)')
ax2.grid(True, alpha=0.3, axis='x')

plt.suptitle('Figure 8a: Episode-Level IMDB Prediction Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figure8a_episode_results.png', dpi=300, bbox_inches='tight')
plt.show()

print('✅ Plot saved as figure8a_episode_results.png')
print(f'Best R² = {max(test_r2):.3f}')
print('Conclusion: Episode-level sentiment cannot predict IMDB ratings') 