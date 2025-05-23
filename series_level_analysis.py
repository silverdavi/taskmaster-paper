#!/usr/bin/env python3
"""
Series-Level Analysis for Figure 8b: IMDB Prediction with Contestant Features

With only 18 series, we need simpler models and more careful feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut, cross_val_score
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("../../data/raw")
RANDOM_STATE = 42

class SeriesLevelAnalyzer:
    """Series-level analysis incorporating contestant features."""
    
    def __init__(self):
        self.target_cols = [f'hist{i}_pct' for i in range(1, 11)]
        
    def load_episode_sentiment_data(self):
        """Load and aggregate episode-level sentiment data by series."""
        print("Loading episode sentiment data...")
        
        # Load sentiment data
        sentiment_file = DATA_DIR / "sentiment.csv"
        sentiment_df = pd.read_csv(sentiment_file)
        
        # Load IMDB data for targets
        imdb_file = DATA_DIR / "taskmaster_histograms_corrected.csv"
        imdb_df = pd.read_csv(imdb_file)
        
        # Create episode identifiers
        sentiment_df['episode_id'] = sentiment_df['series'].astype(str) + '_' + sentiment_df['episode'].astype(str)
        imdb_df['episode_id'] = imdb_df['season'].astype(str) + '_' + imdb_df['episode'].astype(str)
        
        # Merge sentiment and IMDB data
        merged = imdb_df.merge(sentiment_df, on='episode_id', how='inner')
        
        print(f"  Merged data: {len(merged)} episodes")
        
        # Aggregate by series
        sentiment_cols = [
            'avg_anger', 'avg_awkwardness', 'avg_frustration_or_despair',
            'avg_humor', 'avg_joy_or_excitement', 'avg_sarcasm', 'avg_self_deprecation'
        ]
        
        # Aggregate sentiment features (mean across episodes in series)
        series_sentiment = merged.groupby('season')[sentiment_cols].mean().reset_index()
        series_sentiment.rename(columns={'season': 'series'}, inplace=True)
        
        # Aggregate IMDB targets (mean across episodes in series) 
        series_imdb = merged.groupby('season')[self.target_cols].mean().reset_index()
        series_imdb.rename(columns={'season': 'series'}, inplace=True)
        
        # Count episodes per series
        episode_counts = merged.groupby('season').size().reset_index(name='num_episodes')
        episode_counts.rename(columns={'season': 'series'}, inplace=True)
        
        # Merge all series-level data
        series_data = series_imdb.merge(series_sentiment, on='series')
        series_data = series_data.merge(episode_counts, on='series')
        
        print(f"  Series with complete data: {len(series_data)}")
        print(f"  Series range: {series_data['series'].min()}-{series_data['series'].max()}")
        
        return series_data
    
    def load_contestant_data(self):
        """Load and process contestant information."""
        print("Loading contestant data...")
        
        contestants_file = DATA_DIR / "contestants.csv"
        contestants_df = pd.read_csv(contestants_file)
        
        print(f"  Contestants loaded: {len(contestants_df)}")
        
        # Create contestant features by series
        series_features = []
        
        for series_num in contestants_df['series'].unique():
            if pd.isna(series_num):
                continue
                
            series_contestants = contestants_df[contestants_df['series'] == series_num]
            
            # Basic demographics
            avg_age = series_contestants['age_during_taskmaster'].mean()
            prop_female = (series_contestants['gender'] == 'Female').mean()
            prop_non_binary = (series_contestants['gender'] == 'Non-binary').mean()
            
            # Nationality diversity
            unique_nationalities = series_contestants['nationality'].nunique()
            prop_non_british = (~series_contestants['nationality'].str.contains('British|English|Scottish|Welsh|Irish', na=False)).mean()
            
            # Professional background
            prop_comedians = series_contestants['occupation'].str.contains('Comedian', na=False).mean()
            prop_actors = series_contestants['occupation'].str.contains('Actor|Actress', na=False).mean()
            prop_presenters = series_contestants['occupation'].str.contains('Presenter', na=False).mean()
            
            # Years of experience (handle missing values)
            years_active_values = []
            for years_str in series_contestants['years_active']:
                if pd.notna(years_str) and years_str != 'Unknown':
                    try:
                        # Extract numeric years from various formats
                        if 'Since' in str(years_str):
                            start_year = int(str(years_str).split('Since ')[-1])
                            years = 2024 - start_year
                        elif '-' in str(years_str):
                            start_year = int(str(years_str).split('-')[0])
                            years = 2024 - start_year
                        else:
                            years = float(str(years_str))
                        years_active_values.append(years)
                    except:
                        continue
            
            avg_years_active = np.mean(years_active_values) if years_active_values else np.nan
            
            # Create series feature row
            series_features.append({
                'series': int(series_num),
                'avg_age': avg_age,
                'prop_female': prop_female,
                'prop_non_binary': prop_non_binary,
                'nationality_diversity': unique_nationalities,
                'prop_non_british': prop_non_british,
                'prop_comedians': prop_comedians,
                'prop_actors': prop_actors,
                'prop_presenters': prop_presenters,
                'avg_years_active': avg_years_active,
                'num_contestants': len(series_contestants)
            })
        
        contestant_features_df = pd.DataFrame(series_features)
        
        print(f"  Series with contestant data: {len(contestant_features_df)}")
        
        return contestant_features_df
    
    def merge_series_data(self):
        """Merge all series-level data."""
        print("\nMerging series-level data...")
        
        # Load component datasets
        series_data = self.load_episode_sentiment_data()
        contestant_data = self.load_contestant_data()
        
        # Merge on series
        merged = series_data.merge(contestant_data, on='series', how='inner')
        
        print(f"  Final series count: {len(merged)}")
        print(f"  Series: {sorted(merged['series'].tolist())}")
        
        # Handle missing values in contestant features
        contestant_cols = [col for col in merged.columns if col.startswith(('avg_', 'prop_', 'nationality_', 'num_'))]
        
        print("\nMissing values in contestant features:")
        for col in contestant_cols:
            missing = merged[col].isna().sum()
            if missing > 0:
                print(f"  {col}: {missing} missing")
                # Fill with median for numeric features
                if merged[col].dtype in ['float64', 'int64']:
                    merged[col].fillna(merged[col].median(), inplace=True)
        
        print(f"✅ Series-level dataset ready: {len(merged)} series")
        
        return merged
    
    def prepare_features_targets(self, data):
        """Extract feature matrix and target matrix for series-level analysis."""
        
        # Feature columns (sentiment + contestant features)
        sentiment_cols = [
            'avg_anger', 'avg_awkwardness', 'avg_frustration_or_despair',
            'avg_humor', 'avg_joy_or_excitement', 'avg_sarcasm', 'avg_self_deprecation'
        ]
        
        contestant_cols = [
            'avg_age', 'prop_female', 'prop_non_binary', 'nationality_diversity',
            'prop_non_british', 'prop_comedians', 'prop_actors', 'prop_presenters',
            'avg_years_active', 'num_contestants', 'num_episodes'
        ]
        
        # Filter for available columns
        available_sentiment = [col for col in sentiment_cols if col in data.columns]
        available_contestant = [col for col in contestant_cols if col in data.columns]
        
        all_features = available_sentiment + available_contestant
        
        # Extract features and targets
        X = data[all_features].values
        y = data[self.target_cols].values
        
        # Handle any remaining NaN values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        print(f"\nFeature matrix: {X.shape}")
        print(f"Target matrix: {y.shape}")
        print(f"\nFeatures used:")
        for i, feature in enumerate(all_features):
            print(f"  {i+1:2d}. {feature}")
        
        return X, y, all_features
    
    def run_simple_models(self, X, y, feature_names):
        """Run simple models suitable for small datasets (18 samples) with robust NaN handling."""
        print(f"\n{'='*50}")
        print("SERIES-LEVEL MODELING (N=18)")
        print(f"{'='*50}")
        
        # Initialize models suitable for small datasets
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=RANDOM_STATE),
            'Random Forest': RandomForestRegressor(
                n_estimators=50,  # Fewer trees for small dataset
                max_depth=3,      # Shallow trees
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=RANDOM_STATE
            )
        }
        
        results = {}
        
        # Use multiple CV strategies for robustness with small datasets
        from sklearn.model_selection import KFold
        
        # With only 18 samples, use 3-fold CV instead of LOO to avoid single-sample issues
        cv_strategies = {
            'KFold-3': KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            'KFold-5': KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            'Leave-One-Out': LeaveOneOut()
        }
        
        for model_name, model in models.items():
            print(f"\n🔸 {model_name}")
            
            model_results = {}
            
            # Try different CV strategies
            for cv_name, cv_strategy in cv_strategies.items():
                print(f"  📊 {cv_name}:")
                
                try:
                    # Cross-validation scores for each output
                    cv_r2_scores = []
                    cv_mae_scores = []
                    
                    # For each IMDB rating (1-10)
                    for i in range(y.shape[1]):
                        y_single = y[:, i]
                        
                        # Check for constant target values (causes R² issues)
                        if np.std(y_single) < 1e-10:
                            print(f"    ⚠️  Output {i+1}: Target values are constant, skipping")
                            continue
                        
                        # Cross-validation with error handling
                        try:
                            r2_scores = cross_val_score(model, X, y_single, cv=cv_strategy, scoring='r2')
                            mae_scores = -cross_val_score(model, X, y_single, cv=cv_strategy, scoring='neg_mean_absolute_error')
                            
                            # Handle NaN values in cross-validation scores
                            r2_scores_clean = r2_scores[~np.isnan(r2_scores)]
                            mae_scores_clean = mae_scores[~np.isnan(mae_scores)]
                            
                            if len(r2_scores_clean) == 0:
                                print(f"    ⚠️  Output {i+1}: All R² scores are NaN, using baseline (-1.0)")
                                r2_mean = -1.0  # Worse than baseline
                            else:
                                r2_mean = np.mean(r2_scores_clean)
                            
                            if len(mae_scores_clean) == 0:
                                print(f"    ⚠️  Output {i+1}: All MAE scores are NaN, using target std")
                                mae_mean = np.std(y_single)  # Fallback to standard deviation
                            else:
                                mae_mean = np.mean(mae_scores_clean)
                            
                            cv_r2_scores.append(r2_mean)
                            cv_mae_scores.append(mae_mean)
                            
                        except Exception as e:
                            print(f"    ❌ Output {i+1}: CV failed ({str(e)}), using baseline")
                            cv_r2_scores.append(-1.0)  # Baseline fallback
                            cv_mae_scores.append(np.std(y_single))  # Std dev fallback
                    
                    # Calculate overall metrics with NaN handling
                    if len(cv_r2_scores) == 0:
                        avg_r2 = -1.0
                        avg_mae = np.mean([np.std(y[:, i]) for i in range(y.shape[1])])
                        print(f"    ❌ No valid outputs, using baseline values")
                    else:
                        # Replace any remaining NaN values with baseline
                        cv_r2_scores = [score if not np.isnan(score) else -1.0 for score in cv_r2_scores]
                        cv_mae_scores = [score if not np.isnan(score) else np.std(y[:, i]) for i, score in enumerate(cv_mae_scores)]
                        
                        avg_r2 = np.mean(cv_r2_scores)
                        avg_mae = np.mean(cv_mae_scores)
                    
                    print(f"    Cross-validation R²: {avg_r2:.4f}")
                    print(f"    Cross-validation MAE: {avg_mae:.4f}")
                    
                    if len(cv_r2_scores) > 1:
                        print(f"    R² range per output: {min(cv_r2_scores):.3f} - {max(cv_r2_scores):.3f}")
                    
                    model_results[cv_name] = {
                        'cv_r2': avg_r2,
                        'cv_mae': avg_mae,
                        'cv_r2_per_output': cv_r2_scores,
                        'cv_mae_per_output': cv_mae_scores,
                        'valid_outputs': len(cv_r2_scores)
                    }
                    
                except Exception as e:
                    print(f"    ❌ {cv_name} failed completely: {str(e)}")
                    model_results[cv_name] = {
                        'cv_r2': -1.0,
                        'cv_mae': np.mean([np.std(y[:, i]) for i in range(y.shape[1])]),
                        'cv_r2_per_output': [-1.0] * y.shape[1],
                        'cv_mae_per_output': [np.std(y[:, i]) for i in range(y.shape[1])],
                        'valid_outputs': 0
                    }
            
            # Use the most reliable CV strategy (KFold-3 preferred for small datasets)
            if 'KFold-3' in model_results and model_results['KFold-3']['valid_outputs'] > 0:
                best_cv = 'KFold-3'
            elif 'KFold-5' in model_results and model_results['KFold-5']['valid_outputs'] > 0:
                best_cv = 'KFold-5'
            else:
                best_cv = list(model_results.keys())[0]  # Use whatever we have
            
            print(f"  🏆 Using {best_cv} results as primary metric")
            
            # Fit on full data for feature importance (if available)
            try:
                if hasattr(model, 'feature_importances_'):
                    model.fit(X, y)
                    feature_importance = model.feature_importances_
                    
                    # Handle NaN in feature importance
                    if np.any(np.isnan(feature_importance)):
                        print(f"  ⚠️  Some feature importances are NaN, replacing with 0")
                        feature_importance = np.nan_to_num(feature_importance, nan=0.0)
                    
                    print(f"  Top features:")
                    importance_pairs = list(zip(feature_names, feature_importance))
                    importance_pairs.sort(key=lambda x: x[1], reverse=True)
                    
                    for i, (feature, importance) in enumerate(importance_pairs[:5]):
                        print(f"    {i+1}. {feature}: {importance:.4f}")
                        
                    model_results['feature_importance'] = dict(importance_pairs)
                        
            except Exception as e:
                print(f"  ⚠️  Feature importance calculation failed: {str(e)}")
                model_results['feature_importance'] = {name: 0.0 for name in feature_names}
            
            # Store best results for this model
            results[model_name] = model_results[best_cv].copy()
            results[model_name]['all_cv_results'] = model_results
        
        return results
    
    def analyze_series_level(self):
        """Run complete series-level analysis."""
        print("="*60)
        print("FIGURE 8B: SERIES-LEVEL IMDB PREDICTION ANALYSIS")
        print("WITH CONTESTANT FEATURES")
        print("="*60)
        
        # Load and merge data
        series_data = self.merge_series_data()
        
        # Prepare features and targets
        X, y, feature_names = self.prepare_features_targets(series_data)
        
        # Run models
        results = self.run_simple_models(X, y, feature_names)
        
        # Summary with robust NaN handling
        print(f"\n{'='*50}")
        print("SERIES-LEVEL ANALYSIS SUMMARY")
        print(f"{'='*50}")
        
        # Find best model with NaN handling
        valid_results = {}
        for model_name, model_result in results.items():
            r2_score = model_result.get('cv_r2', -1.0)
            if not np.isnan(r2_score):
                valid_results[model_name] = r2_score
            else:
                print(f"⚠️  {model_name}: R² is NaN, treating as -1.0")
                valid_results[model_name] = -1.0
                results[model_name]['cv_r2'] = -1.0  # Fix NaN in results
        
        if valid_results:
            best_model = max(valid_results.keys(), key=lambda k: valid_results[k])
            best_r2 = valid_results[best_model]
        else:
            best_model = "None"
            best_r2 = -1.0
            print("❌ All models failed - no valid R² scores")
        
        print(f"🏆 Best model: {best_model}")
        print(f"📈 Cross-validation R²: {best_r2:.4f}")
        
        # Interpretation with NaN-aware logic
        if np.isnan(best_r2):
            print("❌ Best R² is NaN - analysis failed due to data limitations")
            interpretation = "Analysis failed: Dataset too small or problematic for reliable modeling."
            status_emoji = "❌"
        elif best_r2 > 0.2:
            print("✅ Series-level features show meaningful predictive power!")
            interpretation = "Contestant demographics and sentiment can partially predict series satisfaction."
            status_emoji = "✅"
        elif best_r2 > 0:
            print("⚠️  Series-level features show weak but positive predictive power.")
            interpretation = "Limited predictive power from available features."
            status_emoji = "⚠️"
        elif best_r2 > -0.5:
            print("⚠️  Series-level features show poor predictive power.")
            interpretation = "Current features provide little predictive value."
            status_emoji = "⚠️"
        else:
            print("❌ Series-level features fail to predict IMDB ratings.")
            interpretation = "Current features insufficient for predicting series satisfaction."
            status_emoji = "❌"
        
        print(f"\n🔬 Scientific interpretation:")
        print(f"   {interpretation}")
        
        print(f"\n📊 Dataset summary:")
        print(f"   Series analyzed: {len(series_data)}")
        print(f"   Features used: {len(feature_names)}")
        print(f"   Feature types: Sentiment + Contestant demographics")
        
        # Show all model results for transparency
        print(f"\n📈 All model performance:")
        for model_name, model_result in results.items():
            r2_score = model_result.get('cv_r2', -1.0)
            mae_score = model_result.get('cv_mae', np.nan)
            
            # Handle NaN values in display
            r2_display = f"{r2_score:.4f}" if not np.isnan(r2_score) else "NaN"
            mae_display = f"{mae_score:.4f}" if not np.isnan(mae_score) else "NaN"
            
            print(f"   {model_name}: R² = {r2_display}, MAE = {mae_display}")
        
        # Create comprehensive results summary
        analysis_summary = {
            'status': status_emoji,
            'best_model': best_model,
            'best_r2': best_r2 if not np.isnan(best_r2) else -1.0,
            'interpretation': interpretation,
            'dataset_size': len(series_data),
            'feature_count': len(feature_names),
            'all_model_results': results
        }
        
        return {
            'data': series_data,
            'results': results,
            'features': feature_names,
            'X': X,
            'y': y,
            'summary': analysis_summary
        }

def main():
    """Main analysis function."""
    analyzer = SeriesLevelAnalyzer()
    return analyzer.analyze_series_level()

if __name__ == "__main__":
    series_results = main() 