#!/usr/bin/env python3
"""
Stage 3: Episode-Level Modeling
Train models using selected features for IMDB prediction.
"""

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# Configuration
EPISODE_DATA_FILE = "episode_data.csv"
FEATURE_SELECTION_FILE = "episode_selected_features.json"
OUTPUT_FILE = "episode_model_results.pkl"
RANDOM_STATE = 42

def load_data_and_features():
    """Load episode data and selected features."""
    print("Loading episode data and feature selections...")
    
    # Load episode data
    episode_data = pd.read_csv(EPISODE_DATA_FILE)
    print(f"  Episode data: {episode_data.shape}")
    
    # Load feature selection results
    with open(FEATURE_SELECTION_FILE, 'r') as f:
        feature_selection = json.load(f)
    
    print(f"  Feature selection criteria available:")
    for criteria in feature_selection['selection_criteria']:
        print(f"    - {criteria['name']}: {len(criteria['features'])} features")
    
    return episode_data, feature_selection

def prepare_model_data(episode_data, feature_list):
    """Prepare data for modeling."""
    
    target_cols = [f'hist{i}_pct' for i in range(1, 11)]
    
    # Filter features to only include numeric ones that exist in data
    available_features = []
    for feature in feature_list:
        if feature in episode_data.columns and episode_data[feature].dtype in ['int64', 'float64']:
            available_features.append(feature)
        else:
            print(f"    Warning: Feature '{feature}' not available or not numeric")
    
    print(f"  Using {len(available_features)} features out of {len(feature_list)} requested")
    
    X = episode_data[available_features].values
    y = episode_data[target_cols].values
    
    # Handle any remaining missing values
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    return X, y, available_features

def train_models(X, y, feature_names):
    """Train different models and evaluate performance."""
    print(f"\nTraining models with {X.shape[1]} features on {X.shape[0]} episodes...")
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=RANDOM_STATE),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE
        )
    }
    
    # Split data for final testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    print(f"  Training set: {X_train.shape[0]} episodes")
    print(f"  Test set: {X_test.shape[0]} episodes")
    
    # Scale features for consistency (though only Ridge really benefits)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n  🔸 Training {model_name}...")
        
        # Use scaled features for Ridge, original for Random Forest and Linear Regression
        if model_name == 'Ridge Regression':
            X_train_model = X_train_scaled
            X_test_model = X_test_scaled
        else:
            X_train_model = X_train
            X_test_model = X_test
        
        # Cross-validation on training set
        print(f"    Cross-validation...")
        cv_r2_scores = []
        cv_mae_scores = []
        
        # Cross-validate on each target (IMDB rating 1-10)
        for i in range(y.shape[1]):
            y_single = y_train[:, i]
            
            # Skip if no variance in target
            if np.std(y_single) < 1e-10:
                print(f"      Target {i+1}: No variance, skipping")
                continue
            
            try:
                r2_scores = cross_val_score(model, X_train_model, y_single, 
                                          cv=5, scoring='r2')
                mae_scores = -cross_val_score(model, X_train_model, y_single,
                                            cv=5, scoring='neg_mean_absolute_error')
                
                cv_r2_scores.append(np.mean(r2_scores))
                cv_mae_scores.append(np.mean(mae_scores))
                
            except Exception as e:
                print(f"      Target {i+1}: CV failed ({str(e)})")
                cv_r2_scores.append(-1.0)
                cv_mae_scores.append(np.std(y_single))
        
        avg_cv_r2 = np.mean(cv_r2_scores) if cv_r2_scores else -1.0
        avg_cv_mae = np.mean(cv_mae_scores) if cv_mae_scores else 1.0
        
        print(f"    CV R²: {avg_cv_r2:.4f}")
        print(f"    CV MAE: {avg_cv_mae:.4f}")
        
        # Train on full training set and test
        print(f"    Final training and testing...")
        try:
            model.fit(X_train_model, y_train)
            y_pred = model.predict(X_test_model)
            
            # Calculate test metrics
            test_r2 = r2_score(y_test, y_pred)
            test_mae = mean_absolute_error(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"    Test R²: {test_r2:.4f}")
            print(f"    Test MAE: {test_mae:.4f}")
            print(f"    Test RMSE: {test_rmse:.4f}")
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_names, model.feature_importances_))
                print(f"    Feature importance available")
            elif hasattr(model, 'coef_'):
                # Use coefficient magnitude for linear models
                coef_importance = np.mean(np.abs(model.coef_), axis=0)
                feature_importance = dict(zip(feature_names, coef_importance))
                print(f"    Coefficient importance available")
                
        except Exception as e:
            print(f"    ❌ Training/testing failed: {str(e)}")
            test_r2 = -1.0
            test_mae = 1.0
            test_rmse = 1.0
            y_pred = np.zeros_like(y_test)
            feature_importance = {}
        
        # Store results
        results[model_name] = {
            'cv_r2': avg_cv_r2,
            'cv_mae': avg_cv_mae,
            'cv_r2_per_target': cv_r2_scores,
            'cv_mae_per_target': cv_mae_scores,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'predictions': y_pred,
            'feature_importance': feature_importance or {}
        }
    
    return results, X_test, y_test, scaler

def evaluate_feature_sets(episode_data, feature_selection):
    """Evaluate different feature sets."""
    print("="*60)
    print("EPISODE-LEVEL MODELING WITH FEATURE SELECTION")
    print("="*60)
    
    all_results = {}
    
    # Test different feature selection criteria
    feature_sets_to_test = ['top_5', 'high_mi', 'all_features']
    
    for feature_set_name in feature_sets_to_test:
        print(f"\n{'='*40}")
        print(f"TESTING FEATURE SET: {feature_set_name.upper()}")
        print(f"{'='*40}")
        
        # Get feature list for this criteria
        feature_set = None
        for criteria in feature_selection['selection_criteria']:
            if criteria['name'] == feature_set_name:
                feature_set = criteria
                break
        
        if not feature_set:
            print(f"❌ Feature set '{feature_set_name}' not found")
            continue
            
        features = feature_set['features']
        print(f"Description: {feature_set['description']}")
        print(f"Features ({len(features)}):")
        for i, feat in enumerate(features, 1):
            score = feature_selection['feature_ranking'].get(feat, {}).get('score', 0)
            print(f"  {i:2d}. {feat} (score: {score:.4f})")
        
        # Prepare data
        X, y, available_features = prepare_model_data(episode_data, features)
        
        if len(available_features) == 0:
            print(f"❌ No valid features available for {feature_set_name}")
            continue
        
        # Train models
        results, X_test, y_test, scaler = train_models(X, y, available_features)
        
        # Store results with metadata
        all_results[feature_set_name] = {
            'description': feature_set['description'],
            'features_requested': features,
            'features_used': available_features,
            'results': results,
            'data_shape': {'X': X.shape, 'y': y.shape},
            'test_data': {'X_test': X_test, 'y_test': y_test},
            'scaler': scaler
        }
    
    return all_results

def summarize_results(all_results):
    """Summarize and compare results across feature sets."""
    print(f"\n{'='*60}")
    print("EPISODE-LEVEL MODELING SUMMARY")
    print(f"{'='*60}")
    
    # Find best overall result
    best_score = -np.inf
    best_combination = None
    
    summary_table = []
    
    for feature_set_name, feature_results in all_results.items():
        print(f"\n🔸 Feature Set: {feature_set_name.upper()}")
        print(f"   Description: {feature_results['description']}")
        print(f"   Features used: {len(feature_results['features_used'])}")
        
        for model_name, model_results in feature_results['results'].items():
            test_r2 = model_results['test_r2']
            test_mae = model_results['test_mae']
            cv_r2 = model_results['cv_r2']
            
            print(f"   {model_name:15} | Test R²: {test_r2:7.4f} | CV R²: {cv_r2:7.4f} | MAE: {test_mae:.4f}")
            
            # Track best result
            if test_r2 > best_score:
                best_score = test_r2
                best_combination = (feature_set_name, model_name)
            
            summary_table.append({
                'feature_set': feature_set_name,
                'model': model_name,
                'test_r2': test_r2,
                'cv_r2': cv_r2,
                'test_mae': test_mae,
                'num_features': len(feature_results['features_used'])
            })
    
    print(f"\n🏆 BEST RESULT:")
    if best_combination:
        best_fs, best_model = best_combination
        print(f"   Feature Set: {best_fs}")
        print(f"   Model: {best_model}")
        print(f"   Test R²: {best_score:.4f}")
        
        # Show feature importance for best model
        best_importance = all_results[best_fs]['results'][best_model]['feature_importance']
        if best_importance:
            print(f"\n   Top features:")
            sorted_importance = sorted(best_importance.items(), key=lambda x: x[1], reverse=True)
            for i, (feat, importance) in enumerate(sorted_importance[:5], 1):
                print(f"     {i}. {feat}: {importance:.4f}")
    else:
        print("   No valid results found")
    
    # Overall interpretation
    if best_score > 0.3:
        status = "✅ GOOD"
        interpretation = "Episode features show meaningful predictive power for IMDB ratings"
    elif best_score > 0.1:
        status = "⚠️  MODERATE"
        interpretation = "Episode features show weak but positive predictive power"
    elif best_score > 0:
        status = "⚠️  POOR"
        interpretation = "Episode features show minimal predictive power"
    else:
        status = "❌ FAILED"
        interpretation = "Episode features cannot predict IMDB ratings"
    
    print(f"\n📊 OVERALL ASSESSMENT: {status}")
    print(f"   {interpretation}")
    print(f"   Best R² = {best_score:.4f}")
    
    return summary_table, best_combination

def main():
    """Main modeling function."""
    # Load data and features
    episode_data, feature_selection = load_data_and_features()
    
    # Evaluate different feature sets
    all_results = evaluate_feature_sets(episode_data, feature_selection)
    
    # Summarize results
    summary_table, best_combination = summarize_results(all_results)
    
    # Save comprehensive results
    final_results = {
        'all_results': all_results,
        'summary_table': summary_table,
        'best_combination': best_combination,
        'data_info': {
            'episodes': len(episode_data),
            'targets': 10,
            'total_features_available': len([col for col in episode_data.columns 
                                           if episode_data[col].dtype in ['int64', 'float64']])
        }
    }
    
    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(final_results, f)
    
    print(f"✅ Episode-level modeling completed!")
    print(f"   Results saved to: {OUTPUT_FILE}")
    
    return final_results

if __name__ == "__main__":
    results = main() 