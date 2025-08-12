import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_prepare_data(filepath):
    """Load CSV and prepare data for regression analysis
    Uses T1 labels for client (C/S) and T2 labels for counsellor strategies"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Group by conversation
    conversations = df.groupby('conv_id')
    
    regression_data = []
    
    for conv_id, conv_df in conversations:
        # Calculate change fraction using T1 labels for clients
        client_df = conv_df[conv_df['speaker'] == 'client']
        
        # T1 labels: C = Change, S = Sustain
        change_count = (client_df['t1_label_auto'] == 'C').sum()
        sustain_count = (client_df['t1_label_auto'] == 'S').sum()
        
        total = change_count + sustain_count
        if total > 0:
            change_fraction = change_count / total
            
            # Count T2 BC strategies used by counsellor
            counsellor_df = conv_df[conv_df['speaker'] == 'counsellor']
            
            strategy_counts = {}
            
            # Count T2 strategies only (excluding FI and N which are non-strategies)
            t2_strategies = counsellor_df['t2_label_auto'].value_counts()
            for strategy, count in t2_strategies.items():
                if strategy not in ['FI', 'N', None] and pd.notna(strategy):
                    strategy_counts[f'{strategy}'] = count
            
            regression_data.append({
                'conv_id': conv_id,
                'change_fraction': change_fraction,
                'change_count': change_count,
                'sustain_count': sustain_count,
                **strategy_counts
            })
    
    # Convert to DataFrame and fill NaN with 0
    reg_df = pd.DataFrame(regression_data)
    
    # Remove conversations with no change/sustain data
    initial_count = len(reg_df)
    reg_df = reg_df[reg_df['change_count'] + reg_df['sustain_count'] > 0]
    
    # Fill NaN with 0 for strategy counts
    strategy_cols = [col for col in reg_df.columns if col not in ['conv_id', 'change_fraction', 'change_count', 'sustain_count']]
    reg_df[strategy_cols] = reg_df[strategy_cols].fillna(0)
    
    print(f"Prepared {len(reg_df)} conversations for analysis (from {initial_count} total)")
    print(f"Found {len(strategy_cols)} unique T2 counsellor strategies")
    print(f"Using T1 labels (C/S) for client change/sustain classification")
    
    return reg_df

def perform_individual_regression(reg_df):
    """Perform linear regression for each individual T2 counsellor strategy"""
    print("\n" + "="*60)
    print("T2 COUNSELLOR STRATEGY REGRESSION RESULTS")
    print("(Predicting client change fraction from T1 labels)")
    print("="*60)
    
    y = reg_df['change_fraction'].values
    strategy_cols = [col for col in reg_df.columns if col not in ['conv_id', 'change_fraction', 'change_count', 'sustain_count']]
    
    results = []
    
    for strategy in strategy_cols:
        X = reg_df[strategy].values.reshape(-1, 1)
        
        # Skip if no variation
        if len(np.unique(X)) < 2:
            continue
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate statistics
        r2 = r2_score(y, y_pred)
        n = len(y)
        
        # Calculate standard errors and p-values
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - 2)
        
        # Standard error of coefficient
        X_mean = np.mean(X)
        se_coef = np.sqrt(mse / np.sum((X - X_mean)**2))
        
        # T-statistic and p-value
        t_stat = model.coef_[0] / se_coef
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))
        
        results.append({
            'strategy': strategy,
            'coefficient': model.coef_[0],
            'intercept': model.intercept_,
            'std_error': se_coef,
            'r_squared': r2,
            'p_value': p_value,
            'avg_use': np.mean(X),
            'max_use': np.max(X),
            'n_conversations': np.sum(X > 0),  # Number of conversations using this strategy
            'n': n
        })
    
    # Convert to DataFrame and sort by p-value
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_value')
    
    # Display significant results
    significant = results_df[results_df['p_value'] < 0.05]
    
    print(f"\nStatistically significant T2 strategies (p < 0.05): {len(significant)}")
    print("\n{:<12} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8}".format(
        "Strategy", "Coef", "Std Err", "p-value", "R²", "Avg Use", "N Conv"
    ))
    print("-" * 80)
    
    for _, row in significant.iterrows():
        p_str = "<0.001" if row['p_value'] < 0.001 else f"{row['p_value']:.4f}"
        print(f"{row['strategy']:<12} {row['coefficient']:>10.4f} {row['std_error']:>10.4f} "
              f"{p_str:>10} {row['r_squared']:>8.3f} {row['avg_use']:>8.1f} {row['n_conversations']:>8.0f}")
    
    # Also show top strategies by R² regardless of p-value
    print("\n" + "-"*80)
    print("Top 10 T2 strategies by R² value:")
    print("\n{:<12} {:>10} {:>10} {:>8} {:>8}".format(
        "Strategy", "Coef", "p-value", "R²", "Avg Use"
    ))
    print("-" * 55)
    
    top_r2 = results_df.nlargest(10, 'r_squared')
    for _, row in top_r2.iterrows():
        p_str = "<0.001" if row['p_value'] < 0.001 else f"{row['p_value']:.4f}"
        print(f"{row['strategy']:<12} {row['coefficient']:>10.4f} {p_str:>10} "
              f"{row['r_squared']:>8.3f} {row['avg_use']:>8.1f}")
    
    return results_df

def analyze_strategy_pairs(reg_df, top_n=10):
    """Analyze combinations of T2 counsellor strategies for complementary effects"""
    print("\n" + "="*60)
    print("COMPLEMENTARY T2 STRATEGY PAIRS ANALYSIS")
    print("="*60)
    
    y = reg_df['change_fraction'].values
    strategy_cols = [col for col in reg_df.columns if col not in ['conv_id', 'change_fraction', 'change_count', 'sustain_count']]
    
    # First get individual R² and models for each strategy
    individual_models = {}
    individual_r2 = {}
    for strategy in strategy_cols:
        X = reg_df[strategy].values.reshape(-1, 1)
        if len(np.unique(X)) < 2:
            continue
        model = LinearRegression()
        model.fit(X, y)
        individual_models[strategy] = model
        individual_r2[strategy] = r2_score(y, model.predict(X))
    
    # Get top strategies by R²
    top_strategies = sorted(individual_r2.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_strategy_names = [s[0] for s in top_strategies]
    
    print(f"\nAnalyzing pairs from top {top_n} T2 strategies...")
    
    pair_results = []
    
    for strat1, strat2 in combinations(top_strategy_names, 2):
        # Get individual R² values
        r2_1 = individual_r2[strat1]
        r2_2 = individual_r2[strat2]
        
        # Fit combined model
        X_combined = reg_df[[strat1, strat2]].values
        model_combined = LinearRegression()
        model_combined.fit(X_combined, y)
        y_pred_combined = model_combined.predict(X_combined)
        r2_combined = r2_score(y, y_pred_combined)
        
        # Calculate statistics
        n = len(y)
        k = 2  # number of predictors
        
        # Adjusted R²
        adj_r2 = 1 - ((1 - r2_combined) * (n - 1) / (n - k - 1))
        
        # Calculate residual sum of squares for combined model
        ss_res_combined = np.sum((y - y_pred_combined) ** 2)
        mse_combined = ss_res_combined / (n - k - 1)
        
        # Calculate standard errors for coefficients
        X_with_intercept = np.column_stack([np.ones(n), X_combined])
        XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se_coefs = np.sqrt(mse_combined * np.diag(XtX_inv))
        
        # T-statistics and p-values for individual coefficients
        t_stat_1 = model_combined.coef_[0] / se_coefs[1]
        t_stat_2 = model_combined.coef_[1] / se_coefs[2]
        p_value_1 = 2 * (1 - stats.t.cdf(np.abs(t_stat_1), n - k - 1))
        p_value_2 = 2 * (1 - stats.t.cdf(np.abs(t_stat_2), n - k - 1))
        
        # F-statistic for overall model
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_reg = ss_total - ss_res_combined
        f_stat = (ss_reg / k) / (ss_res_combined / (n - k - 1))
        f_p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
        
        # Test if adding second variable significantly improves model
        # Compare nested models using partial F-test
        X_single = reg_df[strat1].values.reshape(-1, 1)
        model_single = LinearRegression()
        model_single.fit(X_single, y)
        y_pred_single = model_single.predict(X_single)
        ss_res_single = np.sum((y - y_pred_single) ** 2)
        
        # Partial F-test for improvement
        f_improvement = ((ss_res_single - ss_res_combined) / 1) / (ss_res_combined / (n - k - 1))
        p_improvement = 1 - stats.f.cdf(f_improvement, 1, n - k - 1)
        
        # Calculate improvement over best individual
        best_individual = max(r2_1, r2_2)
        improvement = r2_combined - best_individual
        
        pair_results.append({
            'strategy1': strat1,
            'strategy2': strat2,
            'r2_1': r2_1,
            'r2_2': r2_2,
            'r2_combined': r2_combined,
            'adj_r2_combined': adj_r2,
            'improvement': improvement,
            'coef1': model_combined.coef_[0],
            'coef2': model_combined.coef_[1],
            'se_coef1': se_coefs[1],
            'se_coef2': se_coefs[2],
            'p_value_1': p_value_1,
            'p_value_2': p_value_2,
            'f_stat': f_stat,
            'f_p_value': f_p_value,
            'p_improvement': p_improvement
        })
    
    # Sort by improvement
    pair_results_df = pd.DataFrame(pair_results)
    pair_results_df = pair_results_df.sort_values('improvement', ascending=False)
    
    print("\nTop 15 complementary T2 pairs (by R² improvement):")
    print("\n{:<12} {:<12} {:>8} {:>8} {:>10} {:>12}".format(
        "Strategy 1", "Strategy 2", "R²(1)", "R²(2)", "R²(both)", "Improvement"
    ))
    print("-" * 75)
    
    for _, row in pair_results_df.head(15).iterrows():
        print(f"{row['strategy1']:<12} {row['strategy2']:<12} "
              f"{row['r2_1']:>8.3f} {row['r2_2']:>8.3f} "
              f"{row['r2_combined']:>10.3f} {row['improvement']:>+12.4f}")
    
    # Also show pairs with highest combined R² (not just improvement)
    print("\n" + "-"*75)
    print("Top 10 pairs by combined R² value:")
    print("\n{:<12} {:<12} {:>10} {:>10} {:>10}".format(
        "Strategy 1", "Strategy 2", "R²(both)", "Adj R²", "Coef 1", "Coef 2"
    ))
    print("-" * 65)
    
    top_combined = pair_results_df.nlargest(10, 'r2_combined')
    for _, row in top_combined.iterrows():
        print(f"{row['strategy1']:<12} {row['strategy2']:<12} "
              f"{row['r2_combined']:>10.3f} {row['adj_r2_combined']:>10.3f} "
              f"{row['coef1']:>10.4f} {row['coef2']:>10.4f}")
    
    return pair_results_df

def create_visualizations(reg_df, results_df, pair_results_df):
    """Create visualization plots for T2 counsellor strategies"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Top strategies by R²
    top_10 = results_df.nsmallest(10, 'p_value')
    colors = ['green' if p < 0.05 else 'gray' for p in top_10['p_value']]
    
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(top_10)), top_10['r_squared'], color=colors)
    ax1.set_xticks(range(len(top_10)))
    ax1.set_xticklabels(top_10['strategy'], rotation=45, ha='right')
    ax1.set_ylabel('R² Value')
    ax1.set_title('Top 10 T2 Counsellor Strategies by Statistical Significance')
    ax1.grid(True, alpha=0.3)
    
    # Add p-values as text
    for i, (bar, p) in enumerate(zip(bars, top_10['p_value'])):
        p_text = '<0.001' if p < 0.001 else f'{p:.3f}'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'p={p_text}', ha='center', va='bottom', fontsize=8)
    
    # 2. Scatter plot for best strategy
    best_strategy = results_df.iloc[0]['strategy']
    ax2 = axes[0, 1]
    X = reg_df[best_strategy].values
    y = reg_df['change_fraction'].values
    
    ax2.scatter(X, y, alpha=0.5)
    
    # Add regression line
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    X_line = np.linspace(X.min(), X.max(), 100)
    y_pred = model.predict(X_line.reshape(-1, 1))
    ax2.plot(X_line, y_pred, 'r-', lw=2, label='Regression line')
    
    ax2.set_xlabel(f'{best_strategy} Count')
    ax2.set_ylabel('Client Change Fraction (T1: C/(C+S))')
    ax2.set_title(f'Best T2 Strategy: {best_strategy} (R²={results_df.iloc[0]["r_squared"]:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Coefficient plot for significant strategies
    significant = results_df[results_df['p_value'] < 0.05].head(15)
    ax3 = axes[1, 0]
    
    if len(significant) > 0:
        colors = ['red' if coef < 0 else 'blue' for coef in significant['coefficient']]
        ax3.barh(range(len(significant)), significant['coefficient'], color=colors)
        ax3.set_yticks(range(len(significant)))
        ax3.set_yticklabels(significant['strategy'])
        ax3.set_xlabel('Coefficient Value')
        ax3.set_title('Regression Coefficients for Significant T2 Strategies')
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No significant strategies found', ha='center', va='center')
        ax3.set_title('Regression Coefficients')
    
    # 4. Pair improvements
    top_pairs = pair_results_df.head(10)
    ax4 = axes[1, 1]
    
    pair_labels = [f"{row['strategy1']}\n+\n{row['strategy2']}" 
                   for _, row in top_pairs.iterrows()]
    
    colors = ['green' if imp > 0 else 'red' for imp in top_pairs['improvement']]
    ax4.bar(range(len(top_pairs)), top_pairs['improvement'], color=colors)
    ax4.set_xticks(range(len(top_pairs)))
    ax4.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('R² Improvement')
    ax4.set_title('Top T2 Strategy Pairs: R² Improvement over Best Individual')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('T2 Counsellor Strategy Analysis (vs T1 Client Change/Sustain)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('t2_counsellor_strategy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Visualizations saved to 't2_counsellor_strategy_analysis.png'")

def print_strategy_descriptions():
    """Print common T2 strategy code descriptions"""
    print("\n" + "="*60)
    print("T2 COUNSELLOR STRATEGY CODES REFERENCE")
    print("="*60)
    
    descriptions = {
        "OQ": "Open Question",
        "CQ": "Closed Question",
        "SR": "Simple Reflection",
        "AF": "Affirm",
        "SU": "Support",
        "GI": "Give Information",
        "ST": "Structure",
        "FA": "Facilitate",
        "ADW": "Advise With Permission",
        "ADP": "Advise Without Permission",
        "DI": "Direct",
        "RCW": "Raise Concern With Permission",
        "RCP": "Raise Concern Without Permission",
        "EC": "Emphasize Control",
        "WA": "Warn",
        "CO": "Confront",
        "CR": "Change Reflection",
        "AC+": "Affirm Change Talk",
        "AC-": "Affirm Sustain Talk",
        "AB+": "Ability/Desire Positive",
        "AB-": "Ability/Desire Negative",
        "R+": "Reason Positive",
        "R-": "Reason Negative",
        "RF": "Reframe",
        "FI": "Filler (excluded)",
        "N": "Neutral (excluded)"
    }
    
    print("\nCommon T2 counsellor codes:")
    for code, desc in descriptions.items():
        print(f"  {code:<6} = {desc}")
    
    print("\nT1 client codes used for change fraction:")
    print("  C      = Change")
    print("  S      = Sustain")
    print("\nChange fraction = C / (C + S)")

def main():
    """Main execution function"""
    # Load and prepare data
    filepath = 'data/HLQC_annotated.csv'  # Update this path as needed
    reg_df = load_and_prepare_data(filepath)
    
    # Print basic statistics
    print(f"\nDataset Statistics:")
    print(f"  • Total conversations with change/sustain data: {len(reg_df)}")
    print(f"  • Mean change fraction: {reg_df['change_fraction'].mean():.3f}")
    print(f"  • Std change fraction: {reg_df['change_fraction'].std():.3f}")
    print(f"  • Min change fraction: {reg_df['change_fraction'].min():.3f}")
    print(f"  • Max change fraction: {reg_df['change_fraction'].max():.3f}")
    print(f"  • Total change utterances (T1): {reg_df['change_count'].sum():.0f}")
    print(f"  • Total sustain utterances (T1): {reg_df['sustain_count'].sum():.0f}")
    
    # Print strategy descriptions for reference
    print_strategy_descriptions()
    
    # Perform individual regression analysis
    results_df = perform_individual_regression(reg_df)
    
    # Analyze strategy pairs
    pair_results_df = analyze_strategy_pairs(reg_df)
    
    # Create visualizations
    create_visualizations(reg_df, results_df, pair_results_df)
    
    # Save results to CSV
    results_df.to_csv('t2_counsellor_strategy_results.csv', index=False)
    pair_results_df.to_csv('t2_counsellor_pair_results.csv', index=False)
    reg_df.to_csv('conversation_level_data.csv', index=False)
    
    print("\n✓ Results saved to:")
    print("  • t2_counsellor_strategy_results.csv - Individual T2 strategy regression results")
    print("  • t2_counsellor_pair_results.csv - T2 strategy pair combination analysis") 
    print("  • conversation_level_data.csv - Conversation-level data with change fractions")
    
    # Print summary
    print("\n" + "="*60)
    print("KEY FINDINGS SUMMARY")
    print("="*60)
    
    sig_results = results_df[results_df['p_value'] < 0.05]
    if len(sig_results) > 0:
        print(f"\n✓ {len(sig_results)} T2 counsellor strategies significantly predict client change")
        print("\nTop 5 most effective strategies:")
        for i, row in sig_results.head(5).iterrows():
            effect = "increases" if row['coefficient'] > 0 else "decreases"
            print(f"  • {row['strategy']}: {effect} change fraction by {abs(row['coefficient']):.4f} per use")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()