#!/usr/bin/env python3
"""
Stage 2: Episode-Level Feature Selection
Information entropy analysis to select the most informative features.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import json

# Configuration
INPUT_FILE = "episode_data.csv"
OUTPUT_FILE = "episode_selected_features.json"

def load_episode_data():
    """Load prepared episode data."""
    print("Loading episode data...")
    data = pd.read_csv(INPUT_FILE)
    print(f"  Loaded: {data.shape[0]} episodes, {data.shape[1]} columns")
    return data

def compute_mutual_information(X, y, feature_names):
    """Compute mutual information between features and targets."""
    print("\nComputing mutual information...")
    
    # Ensure no missing values
    X_clean = np.nan_to_num(X, nan=0.0)
    y_clean = np.nan_to_num(y, nan=0.0)
    
    # Compute MI for each target variable
    mi_scores = {}
    for i, target_name in enumerate([f'hist{j}_pct' for j in range(1, 11)]):
        y_target = y_clean[:, i]
        
        # Skip if target has no variance
        if np.std(y_target) < 1e-10:
            print(f"  Skipping {target_name}: no variance")
            continue
            
        mi_values = mutual_info_regression(X_clean, y_target, random_state=42)
        mi_scores[target_name] = dict(zip(feature_names, mi_values))
        
        # Show top features for this target
        sorted_features = sorted(mi_scores[target_name].items(), key=lambda x: x[1], reverse=True)
        print(f"  {target_name} - Top 3 features:")
        for feat, score in sorted_features[:3]:
            print(f"    {feat}: {score:.4f}")
    
    return mi_scores

def compute_feature_entropy(X, feature_names):
    """Compute entropy of each feature."""
    print("\nComputing feature entropy...")
    
    feature_entropy = {}
    for i, feature in enumerate(feature_names):
        values = X[:, i]
        
        # Discretize continuous values for entropy calculation
        if np.std(values) > 1e-10:
            # Use histogram to discretize
            hist, _ = np.histogram(values, bins=10)
            # Normalize to probabilities
            probs = hist / np.sum(hist)
            # Remove zero probabilities
            probs = probs[probs > 0]
            # Compute entropy
            feat_entropy = entropy(probs)
        else:
            feat_entropy = 0.0
            
        feature_entropy[feature] = feat_entropy
        
    # Sort by entropy
    sorted_entropy = sorted(feature_entropy.items(), key=lambda x: x[1], reverse=True)
    
    print(f"  Top 5 features by entropy:")
    for feat, ent in sorted_entropy[:5]:
        print(f"    {feat}: {ent:.4f}")
        
    return feature_entropy

def analyze_feature_correlations(X, feature_names):
    """Analyze correlations between features to avoid redundancy."""
    print("\nAnalyzing feature correlations...")
    
    # Convert to DataFrame for easier correlation analysis
    feature_df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = feature_df.corr().abs()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr_value = corr_matrix.iloc[i, j]
            if corr_value > 0.8:  # High correlation threshold
                high_corr_pairs.append((feature_names[i], feature_names[j], corr_value))
    
    if high_corr_pairs:
        print(f"  Found {len(high_corr_pairs)} highly correlated pairs (>0.8):")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"    {feat1} ↔ {feat2}: {corr:.3f}")
    else:
        print("  No highly correlated features found")
        
    return corr_matrix, high_corr_pairs

def select_features():
    """Run feature selection analysis."""
    print("="*60)
    print("EPISODE-LEVEL FEATURE SELECTION")
    print("="*60)
    
    # Load data
    data = load_episode_data()
    
    # Separate features and targets
    target_cols = [f'hist{i}_pct' for i in range(1, 11)]
    feature_cols = [col for col in data.columns 
                   if col not in target_cols + ['episode_id', 'season', 'episode', 'imdb_id']]
    
    print(f"\nAvailable features ({len(feature_cols)}):")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {feat}")
    
    # Filter out non-numeric features
    numeric_feature_cols = []
    for col in feature_cols:
        if data[col].dtype in ['int64', 'float64']:
            numeric_feature_cols.append(col)
        else:
            print(f"  Skipping non-numeric feature: {col} (dtype: {data[col].dtype})")
    
    print(f"\nNumeric features to analyze ({len(numeric_feature_cols)}):")
    for i, feat in enumerate(numeric_feature_cols, 1):
        print(f"  {i:2d}. {feat}")
    
    X = data[numeric_feature_cols].values
    y = data[target_cols].values
    
    # Standardize features for analysis
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Mutual Information Analysis
    mi_scores = compute_mutual_information(X_scaled, y, numeric_feature_cols)
    
    # 2. Feature Entropy Analysis  
    feature_entropy = compute_feature_entropy(X_scaled, numeric_feature_cols)
    
    # 3. Correlation Analysis
    corr_matrix, high_corr_pairs = analyze_feature_correlations(X_scaled, numeric_feature_cols)
    
    # 4. Combine scores for feature ranking
    print(f"\nCombining selection criteria...")
    
    # Average MI across all targets
    avg_mi_scores = {}
    for feature in numeric_feature_cols:
        mi_values = [mi_scores[target].get(feature, 0) for target in mi_scores.keys()]
        avg_mi_scores[feature] = np.mean(mi_values)
    
    # Normalize scores to [0, 1]
    max_mi = max(avg_mi_scores.values()) if avg_mi_scores.values() else 1
    max_entropy = max(feature_entropy.values()) if feature_entropy.values() else 1
    
    combined_scores = {}
    for feature in numeric_feature_cols:
        mi_norm = avg_mi_scores[feature] / max_mi if max_mi > 0 else 0
        entropy_norm = feature_entropy[feature] / max_entropy if max_entropy > 0 else 0
        
        # Combined score: weighted average of MI and entropy
        combined_scores[feature] = 0.7 * mi_norm + 0.3 * entropy_norm
    
    # Sort features by combined score
    ranked_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nFeature ranking (MI: 70%, Entropy: 30%):")
    for i, (feature, score) in enumerate(ranked_features, 1):
        mi_score = avg_mi_scores[feature]
        entropy_score = feature_entropy[feature]
        print(f"  {i:2d}. {feature:<30} | Combined: {score:.4f} | MI: {mi_score:.4f} | Entropy: {entropy_score:.4f}")
    
    # 5. Select top features
    selection_criteria = [
        {"name": "top_5", "description": "Top 5 features by combined score", 
         "features": [feat for feat, _ in ranked_features[:5]]},
        {"name": "top_10", "description": "Top 10 features by combined score", 
         "features": [feat for feat, _ in ranked_features[:10]]},
        {"name": "high_mi", "description": "Features with MI > 0.01", 
         "features": [feat for feat, score in avg_mi_scores.items() if score > 0.01]},
        {"name": "all_features", "description": "All available features", 
         "features": numeric_feature_cols}
    ]
    
    # Remove highly correlated features from selections
    for criteria in selection_criteria:
        if criteria["name"] != "all_features":
            # Remove redundant features
            filtered_features = []
            for feature in criteria["features"]:
                # Check if this feature is highly correlated with any already selected
                is_redundant = False
                for selected_feature in filtered_features:
                    if feature in corr_matrix.columns and selected_feature in corr_matrix.columns:
                        if abs(corr_matrix.loc[feature, selected_feature]) > 0.8:
                            is_redundant = True
                            break
                if not is_redundant:
                    filtered_features.append(feature)
            criteria["features"] = filtered_features
    
    # Display selection results
    print(f"\nFeature selection results:")
    for criteria in selection_criteria:
        features = criteria["features"]
        print(f"\n  {criteria['name'].upper()}: {criteria['description']}")
        print(f"    Selected: {len(features)} features")
        for i, feat in enumerate(features, 1):
            score = combined_scores.get(feat, 0)
            print(f"      {i:2d}. {feat} (score: {score:.4f})")
    
    # 6. Save results
    results = {
        "feature_ranking": {feat: {"rank": i+1, "score": score, "mi": avg_mi_scores[feat], 
                                  "entropy": feature_entropy[feat]} 
                           for i, (feat, score) in enumerate(ranked_features)},
        "selection_criteria": selection_criteria,
        "mutual_info_by_target": mi_scores,
        "high_correlation_pairs": [{"feature1": f1, "feature2": f2, "correlation": corr} 
                                  for f1, f2, corr in high_corr_pairs],
        "summary": {
            "total_features": len(numeric_feature_cols),
            "episodes": len(data),
            "targets": len(target_cols)
        }
    }
    
    print(f"\nSaving feature selection results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Feature selection completed!")
    print(f"   Analyzed: {len(numeric_feature_cols)} features")
    print(f"   Top recommendation: Use 'top_5' or 'high_mi' feature sets")
    
    return results

if __name__ == "__main__":
    results = select_features() 