#!/usr/bin/env python3
"""
Analyze relationships between task properties and score dynamics in Taskmaster UK.

This script examines how different task characteristics (location, type, skills required, etc.)
relate to various score dynamics metrics such as score distribution, tie patterns, and 
statistical measures of score patterns.

Usage:
    python analyze_task_properties_vs_scoring.py

Output:
    - Statistical analysis results in the console
    - Visualizations saved to 'results/property_score_analysis/' directory
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from pathlib import Path

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Create results directory if it doesn't exist
results_dir = Path("results/property_score_analysis")
results_dir.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load and merge task property and score dynamics data."""
    # Load the two primary datasets
    score_metrics = pd.read_csv("data/processed/task/scores_metrics.csv")
    task_properties = pd.read_csv("data/processed/task/tasks_standardized_final.csv")
    
    # Merge datasets on task identifiers
    # We need to create consistent keys for joining the datasets
    score_metrics['task_key'] = score_metrics['series'].astype(str) + "_" + score_metrics['episode'].astype(str) + "_" + score_metrics['task_id'].astype(str)
    task_properties['task_key'] = task_properties['series'].astype(str) + "_" + task_properties['episode'].astype(str) + "_" + task_properties['task_id'].astype(str)
    
    # Merge the datasets
    merged_data = pd.merge(score_metrics, task_properties, on='task_key', how='inner', suffixes=('', '_prop'))
    
    # Drop duplicate columns
    duplicate_cols = [col for col in merged_data.columns if col.endswith('_prop')]
    merged_data.drop(columns=duplicate_cols, inplace=True)
    
    print(f"Loaded {len(merged_data)} tasks with both property and score data")
    
    return merged_data

def preprocess_data(data):
    """Clean and prepare data for analysis."""
    # Convert categorical columns to categorical data type
    categorical_cols = [
        'standardized_location', 'location_category', 'standardized_task_type',
        'standardized_group_solo', 'primary_type'
    ]
    
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype('category')
    
    # Create binary indicators for tie patterns
    data['has_ties'] = data['num_ties'] > 0
    
    # Extract metrics from tie pattern
    data['tie_length'] = data['tie_pattern'].apply(
        lambda x: max([len(group) for group in str(x).split('-')]) if pd.notna(x) else np.nan
    )
    
    # Create feature for score uniformity
    data['score_uniformity'] = 1 - (data['score_std'] / data['score_range'].replace(0, np.nan))
    
    # Calculate score "lumpiness" - how clustered the scores are
    data['score_lumpiness'] = data['num_unique_scores'] / data['num_contestants']
    
    return data

def analyze_location_effects(data):
    """Analyze how task location relates to score dynamics."""
    print("\n--- Analysis of Location Effects on Scoring ---")
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Task Location Effects on Score Dynamics', fontsize=16)
    
    # Analyze number of ties by location category
    sns.barplot(x='location_category', y='num_ties', data=data, ax=axes[0, 0])
    axes[0, 0].set_title('Average Number of Ties by Location')
    axes[0, 0].set_xlabel('Location Category')
    axes[0, 0].set_ylabel('Average Number of Ties')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Analyze score range by location
    sns.boxplot(x='location_category', y='score_range', data=data, ax=axes[0, 1])
    axes[0, 1].set_title('Score Range Distribution by Location')
    axes[0, 1].set_xlabel('Location Category')
    axes[0, 1].set_ylabel('Score Range')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Analyze score entropy by location
    sns.boxplot(x='location_category', y='tie_entropy', data=data, ax=axes[1, 0])
    axes[1, 0].set_title('Tie Entropy by Location')
    axes[1, 0].set_xlabel('Location Category')
    axes[1, 0].set_ylabel('Tie Entropy')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Score distribution by location
    sns.boxplot(x='location_category', y='score_uniformity', data=data, ax=axes[1, 1])
    axes[1, 1].set_title('Score Uniformity by Location')
    axes[1, 1].set_xlabel('Location Category')
    axes[1, 1].set_ylabel('Score Uniformity')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'location_score_effects.png', dpi=300)
    plt.savefig(results_dir / 'location_score_effects.pdf')
    plt.close()
    
    # Statistical tests
    categories = data['location_category'].dropna().unique()
    
    print("\nStatistical Tests for Location Effects:")
    
    # ANOVA tests for numerical score metrics across location categories
    for metric in ['num_ties', 'score_range', 'tie_entropy', 'score_uniformity']:
        groups = [data[data['location_category'] == cat][metric].dropna() for cat in categories]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) >= 2:
            try:
                f_val, p_val = stats.f_oneway(*groups)
                print(f"{metric} ~ location_category: F={f_val:.2f}, p={p_val:.4f}")
            except:
                print(f"Could not perform ANOVA for {metric} due to insufficient data")

def analyze_task_type_effects(data):
    """Analyze how task type relates to score dynamics."""
    print("\n--- Analysis of Task Type Effects on Scoring ---")
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Task Type Effects on Score Dynamics', fontsize=16)
    
    # Plot score standard deviation by task type
    sns.boxplot(x='standardized_task_type', y='score_std', data=data, ax=axes[0, 0])
    axes[0, 0].set_title('Score Standard Deviation by Task Type')
    axes[0, 0].set_xlabel('Task Type')
    axes[0, 0].set_ylabel('Score Standard Deviation')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot number of unique scores by task type
    sns.boxplot(x='standardized_task_type', y='num_unique_scores', data=data, ax=axes[0, 1])
    axes[0, 1].set_title('Number of Unique Scores by Task Type')
    axes[0, 1].set_xlabel('Task Type')
    axes[0, 1].set_ylabel('Number of Unique Scores')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot probability of uniform distribution by task type
    sns.boxplot(x='standardized_task_type', y='chisq_uniform_pval', data=data, ax=axes[1, 0])
    axes[1, 0].set_title('Chi-Square Uniformity p-value by Task Type')
    axes[1, 0].set_xlabel('Task Type')
    axes[1, 0].set_ylabel('Chi-Square p-value')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot tie frequency by task type
    sns.barplot(x='standardized_task_type', y='has_ties', data=data, ax=axes[1, 1])
    axes[1, 1].set_title('Proportion of Tasks with Ties by Task Type')
    axes[1, 1].set_xlabel('Task Type')
    axes[1, 1].set_ylabel('Proportion with Ties')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'task_type_score_effects.png', dpi=300)
    plt.savefig(results_dir / 'task_type_score_effects.pdf')
    plt.close()
    
    # Statistical tests
    task_types = data['standardized_task_type'].dropna().unique()
    
    print("\nStatistical Tests for Task Type Effects:")
    
    # ANOVA tests for numerical score metrics across task types
    for metric in ['score_std', 'num_unique_scores', 'chisq_uniform_pval']:
        groups = [data[data['standardized_task_type'] == task_type][metric].dropna() for task_type in task_types]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) >= 2:
            try:
                f_val, p_val = stats.f_oneway(*groups)
                print(f"{metric} ~ standardized_task_type: F={f_val:.2f}, p={p_val:.4f}")
            except:
                print(f"Could not perform ANOVA for {metric} due to insufficient data")
    
    # Chi-square test for tie presence vs task type
    crosstab = pd.crosstab(data['standardized_task_type'], data['has_ties'])
    if crosstab.shape[0] > 1 and crosstab.shape[1] > 1:
        chi2, p, dof, expected = stats.chi2_contingency(crosstab)
        print(f"has_ties ~ standardized_task_type: Chi2={chi2:.2f}, p={p:.4f}")

def analyze_group_vs_solo_effects(data):
    """Analyze how group vs solo tasks differ in score dynamics."""
    print("\n--- Analysis of Group vs Solo Task Effects on Scoring ---")
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Group vs Solo Task Effects on Score Dynamics', fontsize=16)
    
    # Filter to ensure we only have group and solo tasks
    filtered_data = data[data['standardized_group_solo'].isin(['Group', 'Solo'])]
    
    # Plot score range by group/solo
    sns.boxplot(x='standardized_group_solo', y='score_range', data=filtered_data, ax=axes[0, 0])
    axes[0, 0].set_title('Score Range by Task Format')
    axes[0, 0].set_xlabel('Task Format')
    axes[0, 0].set_ylabel('Score Range')
    
    # Plot tie entropy by group/solo
    sns.boxplot(x='standardized_group_solo', y='tie_entropy', data=filtered_data, ax=axes[0, 1])
    axes[0, 1].set_title('Tie Entropy by Task Format')
    axes[0, 1].set_xlabel('Task Format')
    axes[0, 1].set_ylabel('Tie Entropy')
    
    # Plot score lumpiness by group/solo
    sns.boxplot(x='standardized_group_solo', y='score_lumpiness', data=filtered_data, ax=axes[1, 0])
    axes[1, 0].set_title('Score Lumpiness by Task Format')
    axes[1, 0].set_xlabel('Task Format')
    axes[1, 0].set_ylabel('Score Lumpiness (Unique Scores / Contestants)')
    
    # Plot wasserstein distance by group/solo
    sns.boxplot(x='standardized_group_solo', y='wasserstein_distance_from_perfect', data=filtered_data, ax=axes[1, 1])
    axes[1, 1].set_title('Wasserstein Distance by Task Format')
    axes[1, 1].set_xlabel('Task Format')
    axes[1, 1].set_ylabel('Wasserstein Distance from Perfect')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'group_solo_score_effects.png', dpi=300)
    plt.savefig(results_dir / 'group_solo_score_effects.pdf')
    plt.close()
    
    # Statistical tests
    print("\nStatistical Tests for Group vs Solo Effects:")
    
    # T-tests for numerical metrics between group and solo tasks
    for metric in ['score_range', 'tie_entropy', 'score_lumpiness', 'wasserstein_distance_from_perfect']:
        group_data = filtered_data[filtered_data['standardized_group_solo'] == 'Group'][metric].dropna()
        solo_data = filtered_data[filtered_data['standardized_group_solo'] == 'Solo'][metric].dropna()
        
        if len(group_data) > 0 and len(solo_data) > 0:
            t_stat, p_val = stats.ttest_ind(group_data, solo_data, equal_var=False)
            print(f"{metric} between Group and Solo: t={t_stat:.2f}, p={p_val:.4f}")
            print(f"  Group mean: {group_data.mean():.2f}, Solo mean: {solo_data.mean():.2f}")

def analyze_skill_requirements(data):
    """Analyze how skill requirements relate to score dynamics."""
    print("\n--- Analysis of Skill Requirement Effects on Scoring ---")
    
    # Extract skill scores
    skill_cols = [
        'std_weirdness_score', 'std_creativity_required_score', 
        'std_physical_demand_score', 'std_technical_difficulty_score',
        'std_entertainment_value_score', 'std_time_pressure_score',
        'std_preparation_possible_score', 'std_luck_factor_score'
    ]
    
    # Create a correlation matrix with skill scores and score metrics
    score_metrics = [
        'score_std', 'score_range', 'num_unique_scores', 'tie_entropy',
        'num_ties', 'wasserstein_distance_from_perfect', 'l1_distance_from_perfect'
    ]
    
    # Get only the columns we need for the correlation analysis
    corr_cols = skill_cols + score_metrics
    corr_data = data[corr_cols].copy()
    
    # Compute correlation matrix
    corr_matrix = corr_data.corr()
    
    # Extract only the correlations between skill scores and score metrics
    skill_score_corrs = corr_matrix.loc[skill_cols, score_metrics]
    
    # Plot the correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(skill_score_corrs, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlations between Skill Requirements and Score Dynamics', fontsize=16)
    plt.tight_layout()
    plt.savefig(results_dir / 'skill_score_correlations.png', dpi=300)
    plt.savefig(results_dir / 'skill_score_correlations.pdf')
    plt.close()
    
    # Print top correlations
    print("\nTop Correlations between Skill Requirements and Score Dynamics:")
    # Flatten the correlation matrix into pairs
    corr_pairs = []
    for skill in skill_cols:
        for metric in score_metrics:
            corr_pairs.append((skill, metric, corr_matrix.loc[skill, metric]))
    
    # Sort by absolute correlation and print top results
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for skill, metric, corr in corr_pairs[:10]:
        p_value = stats.pearsonr(data[skill].dropna(), data[metric].dropna())[1]
        print(f"{skill} ~ {metric}: r={corr:.3f}, p={p_value:.4f}")

def regression_analysis(data):
    """Perform regression analysis to predict score dynamics from task properties."""
    print("\n--- Regression Analysis: Predicting Score Dynamics from Task Properties ---")
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score, mean_squared_error
    
    # Define target variables to predict
    target_vars = [
        'score_std', 'tie_entropy', 'num_unique_scores', 
        'wasserstein_distance_from_perfect'
    ]
    
    # Define numerical features
    numerical_features = [
        'std_weirdness_score', 'std_creativity_required_score', 
        'std_physical_demand_score', 'std_technical_difficulty_score',
        'std_entertainment_value_score', 'std_time_pressure_score',
        'std_preparation_possible_score', 'std_luck_factor_score'
    ]
    
    # Define categorical features
    categorical_features = [
        'location_category', 'standardized_task_type', 'standardized_group_solo'
    ]
    
    # Set up preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Results dictionary
    regression_results = {}
    
    # Perform regression for each target variable
    for target in target_vars:
        # Filter out rows with NaN values in the target
        mask = ~data[target].isna()
        target_data = data[mask].copy()
        
        X = target_data[numerical_features + categorical_features]
        y = target_data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        regression_results[target] = {
            'r2': r2,
            'rmse': rmse,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
        
        print(f"\nRegression analysis for {target}:")
        print(f"R² score: {r2:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Save regression results to JSON
    with open(results_dir / 'regression_results.json', 'w') as f:
        json.dump(regression_results, f, indent=2)

def analyze_temporal_patterns(data):
    """Analyze how score dynamics have evolved over the series."""
    print("\n--- Analysis of Temporal Patterns in Score Dynamics ---")
    
    # Group by series
    series_data = data.groupby('series').agg({
        'score_std': 'mean',
        'num_unique_scores': 'mean',
        'tie_entropy': 'mean',
        'num_ties': 'mean',
        'has_ties': 'mean',
        'wasserstein_distance_from_perfect': 'mean',
        'score_range': 'mean'
    }).reset_index()
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Evolution of Score Dynamics Across Series', fontsize=16)
    
    # Plot score standard deviation over time
    sns.lineplot(x='series', y='score_std', data=series_data, marker='o', ax=axes[0, 0])
    axes[0, 0].set_title('Average Score Standard Deviation by Series')
    axes[0, 0].set_xlabel('Series')
    axes[0, 0].set_ylabel('Average Score Std Dev')
    
    # Plot number of unique scores over time
    sns.lineplot(x='series', y='num_unique_scores', data=series_data, marker='o', ax=axes[0, 1])
    axes[0, 1].set_title('Average Number of Unique Scores by Series')
    axes[0, 1].set_xlabel('Series')
    axes[0, 1].set_ylabel('Average Unique Scores')
    
    # Plot tie frequency over time
    sns.lineplot(x='series', y='has_ties', data=series_data, marker='o', ax=axes[1, 0])
    axes[1, 0].set_title('Proportion of Tasks with Ties by Series')
    axes[1, 0].set_xlabel('Series')
    axes[1, 0].set_ylabel('Proportion with Ties')
    
    # Plot Wasserstein distance over time
    sns.lineplot(x='series', y='wasserstein_distance_from_perfect', data=series_data, marker='o', ax=axes[1, 1])
    axes[1, 1].set_title('Average Wasserstein Distance by Series')
    axes[1, 1].set_xlabel('Series')
    axes[1, 1].set_ylabel('Average Wasserstein Distance')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'temporal_score_patterns.png', dpi=300)
    plt.savefig(results_dir / 'temporal_score_patterns.pdf')
    plt.close()
    
    # Test for trends
    print("\nTemporal Trend Analysis:")
    for metric in ['score_std', 'num_unique_scores', 'has_ties', 'wasserstein_distance_from_perfect']:
        result = stats.linregress(series_data['series'], series_data[metric])
        print(f"Linear trend for {metric}: slope={result.slope:.4f}, p={result.pvalue:.4f}")

def analyze_score_metric_dimensions_and_correct_pvalues(data):
    """
    Analyze the actual dimensions (degrees of freedom) in score metrics using PCA,
    and apply appropriate multiple comparison correction to p-values.
    """
    print("\n--- Score Metric Dimensionality Analysis and P-value Correction ---")
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from statsmodels.stats.multitest import multipletests
    
    # Select all score dynamics metrics for PCA
    score_metrics = [
        'score_std', 'score_range', 'score_mean', 'score_median',
        'num_unique_scores', 'tie_entropy', 'num_ties',
        'wasserstein_distance_from_perfect', 'l1_distance_from_perfect',
        'is_permutation', 'chisq_uniform_pval'
    ]
    
    # Filter metrics actually present in the data
    available_metrics = [m for m in score_metrics if m in data.columns]
    
    # Ensure we have enough data points
    score_data = data[available_metrics].dropna()
    print(f"Analyzing {len(available_metrics)} score metrics across {len(score_data)} tasks")
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(score_data)
    
    # Perform PCA
    pca = PCA()
    pca.fit(scaled_data)
    
    # Determine effective dimensions (degrees of freedom)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Number of components needed to explain 95% of variance
    effective_dimensions = np.sum(cumulative_variance < 0.95) + 1
    
    print(f"\nEffective dimensions (DoF) in score metrics: {effective_dimensions}")
    print(f"Components needed to explain 95% of variance: {effective_dimensions}")
    print("\nExplained variance by component:")
    for i, var in enumerate(explained_variance_ratio):
        print(f"  PC{i+1}: {var:.3f} ({cumulative_variance[i]:.3f} cumulative)")
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.6)
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', color='red')
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA: Explained Variance by Component')
    plt.tight_layout()
    plt.savefig(results_dir / 'pca_explained_variance.png', dpi=300)
    plt.savefig(results_dir / 'pca_explained_variance.pdf')
    plt.close()
    
    # Collect p-values from all statistical tests conducted
    print("\n--- Multiple Comparison Correction ---")
    print(f"Using effective DoF: {effective_dimensions} for correction factor")
    
    # Collect p-values from our analyses
    p_values = []
    test_descriptions = []
    
    # Location effects
    categories = data['location_category'].dropna().unique()
    for metric in ['num_ties', 'score_range', 'tie_entropy', 'score_uniformity']:
        groups = [data[data['location_category'] == cat][metric].dropna() for cat in categories]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) >= 2:
            try:
                _, p_val = stats.f_oneway(*groups)
                p_values.append(p_val)
                test_descriptions.append(f"ANOVA: {metric} ~ location_category")
            except:
                pass
    
    # Task type effects
    task_types = data['standardized_task_type'].dropna().unique()
    for metric in ['score_std', 'num_unique_scores', 'chisq_uniform_pval']:
        groups = [data[data['standardized_task_type'] == task_type][metric].dropna() for task_type in task_types]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) >= 2:
            try:
                _, p_val = stats.f_oneway(*groups)
                p_values.append(p_val)
                test_descriptions.append(f"ANOVA: {metric} ~ standardized_task_type")
            except:
                pass
    
    # Chi-square test for task type and ties
    crosstab = pd.crosstab(data['standardized_task_type'], data['has_ties'])
    if crosstab.shape[0] > 1 and crosstab.shape[1] > 1:
        _, p, _, _ = stats.chi2_contingency(crosstab)
        p_values.append(p)
        test_descriptions.append("Chi2: has_ties ~ standardized_task_type")
    
    # Group vs Solo effects
    filtered_data = data[data['standardized_group_solo'].isin(['Group', 'Solo'])]
    for metric in ['score_range', 'tie_entropy', 'score_lumpiness', 'wasserstein_distance_from_perfect']:
        group_data = filtered_data[filtered_data['standardized_group_solo'] == 'Group'][metric].dropna()
        solo_data = filtered_data[filtered_data['standardized_group_solo'] == 'Solo'][metric].dropna()
        
        if len(group_data) > 0 and len(solo_data) > 0:
            _, p_val = stats.ttest_ind(group_data, solo_data, equal_var=False)
            p_values.append(p_val)
            test_descriptions.append(f"t-test: {metric} ~ Group vs Solo")
    
    # Skill correlations
    skill_cols = [
        'std_weirdness_score', 'std_creativity_required_score', 
        'std_physical_demand_score', 'std_technical_difficulty_score',
        'std_entertainment_value_score', 'std_time_pressure_score',
        'std_preparation_possible_score', 'std_luck_factor_score'
    ]
    
    score_metrics_corr = [
        'score_std', 'score_range', 'num_unique_scores', 'tie_entropy',
        'num_ties', 'wasserstein_distance_from_perfect', 'l1_distance_from_perfect'
    ]
    
    for skill in skill_cols:
        for metric in score_metrics_corr:
            if skill in data.columns and metric in data.columns:
                # Fix: Get common indices where both values are non-null
                valid_idx = data[[skill, metric]].dropna().index
                if len(valid_idx) > 0:
                    _, p_val = stats.pearsonr(data.loc[valid_idx, skill], data.loc[valid_idx, metric])
                    p_values.append(p_val)
                    test_descriptions.append(f"Correlation: {skill} ~ {metric}")
    
    # Apply correction for multiple comparisons
    # 1. Bonferroni with effective dimensions
    bonferroni_corrected = np.minimum(np.array(p_values) * effective_dimensions, 1.0)
    
    # 2. FDR (Benjamini-Hochberg)
    rejected, fdr_corrected, _, _ = multipletests(p_values, method='fdr_bh')
    
    # Print results
    print(f"\nUncorrected and corrected p-values for {len(p_values)} statistical tests:")
    print(f"{'Test':<50} {'Uncorrected p':<15} {'Bonferroni p':<15} {'FDR p':<15} {'Significant?':<12}")
    print('-' * 100)
    
    for i, (test, p_orig, p_bonf, p_fdr, rej) in enumerate(zip(test_descriptions, p_values, bonferroni_corrected, fdr_corrected, rejected)):
        sig_symbol = '***' if p_fdr < 0.001 else '**' if p_fdr < 0.01 else '*' if p_fdr < 0.05 else ''
        print(f"{test[:48]:<50} {p_orig:.6f}      {p_bonf:.6f}      {p_fdr:.6f}      {sig_symbol}")
    
    # Save data for future reference
    correction_results = pd.DataFrame({
        'test': test_descriptions,
        'p_value': p_values,
        'bonferroni_corrected': bonferroni_corrected,
        'fdr_corrected': fdr_corrected,
        'significant': rejected
    })
    correction_results.to_csv(results_dir / 'pvalue_corrections.csv', index=False)
    
    # Return effective dimensions
    return effective_dimensions

def main():
    """Main function to execute all analyses."""
    print("Analyzing connections between task properties and score dynamics...")
    
    # Load and preprocess data
    data = load_data()
    data = preprocess_data(data)
    
    # Run analyses
    analyze_location_effects(data)
    analyze_task_type_effects(data)
    analyze_group_vs_solo_effects(data)
    analyze_skill_requirements(data)
    regression_analysis(data)
    analyze_temporal_patterns(data)
    
    # Perform PCA and p-value correction
    analyze_score_metric_dimensions_and_correct_pvalues(data)
    
    print("\nAnalysis complete! Results saved to", results_dir)

if __name__ == "__main__":
    main() 