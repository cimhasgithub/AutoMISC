import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def read_and_preprocess_csv(path):
    """Reads the CSV and creates aggregated BC patterns for T1 and T2."""
    df = pd.read_csv(path)

    volley_t1 = df.groupby("corp_vol_idx")["t1_label_auto"].apply(set).apply(lambda s: ", ".join(sorted(s)))
    volley_t2 = df.groupby("corp_vol_idx")["t2_label_auto"].apply(set).apply(lambda s: ", ".join(sorted(s)))

    df["BC Pattern T1"] = df["corp_vol_idx"].map(volley_t1)
    df["BC Pattern T2"] = df["corp_vol_idx"].map(volley_t2)

    return df

def load_client_data(conv_id):
    """Load client C and S counts from JSON config file."""
    file_path = f"/Users/joeberson/Developer/AutoMISC/data/sampled_clients/client_config_{conv_id:02d}.json"
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data.get('client_C', 0), data.get('client_S', 0)
    except FileNotFoundError:
        print(f"Warning: Could not find file {file_path}")
        return None, None


def calculate_client_change_fraction(conv_id):
    """Calculate client change fraction C/(C+S) for a given conversation."""
    c_count, s_count = load_client_data(conv_id)
    if c_count is None or s_count is None:
        return None
    
    total = c_count + s_count
    if total == 0:
        return None
    
    return c_count / total


def calculate_average_bcs_per_turn(df, conv_id):
    """Calculate average number of unique BCs per counsellor turn in a conversation."""
    # Filter for specific conversation and counsellor utterances
    conv_df = df[(df['conv_id'] == conv_id) & (df['speaker'].str.lower() == 'counsellor')]
    
    if conv_df.empty:
        return None
    
    # Group by volley and get unique BCs per volley (using T1 labels)
    unique_bcs_per_volley = conv_df.groupby('corp_vol_idx')['t1_label_auto'].apply(lambda x: len(set(x)))
    
    # Calculate average unique BCs per turn
    avg_unique_bcs = unique_bcs_per_volley.mean()
    
    return avg_unique_bcs


def count_single_bc_volleys(df, conv_id):
    """Count number of volleys where counsellor used exactly 1 unique BC."""
    # Filter for specific conversation and counsellor utterances
    conv_df = df[(df['conv_id'] == conv_id) & (df['speaker'].str.lower() == 'counsellor')]
    
    if conv_df.empty:
        return 0
    
    # Group by volley and count unique BCs per volley
    unique_bcs_per_volley = conv_df.groupby('corp_vol_idx')['t1_label_auto'].apply(lambda x: len(set(x)))
    
    # Count volleys with exactly 1 unique BC
    single_bc_count = (unique_bcs_per_volley == 1).sum()
    
    return single_bc_count


def count_double_bc_volleys(df, conv_id):
    """Count number of volleys where counsellor used exactly 2 unique BCs."""
    # Filter for specific conversation and counsellor utterances
    conv_df = df[(df['conv_id'] == conv_id) & (df['speaker'].str.lower() == 'counsellor')]
    
    if conv_df.empty:
        return 0
    
    # Group by volley and count unique BCs per volley
    unique_bcs_per_volley = conv_df.groupby('corp_vol_idx')['t1_label_auto'].apply(lambda x: len(set(x)))
    
    # Count volleys with exactly 2 unique BCs
    double_bc_count = (unique_bcs_per_volley == 2).sum()
    
    return double_bc_count


def perform_regression_analysis(df, metric_func, metric_name, plot_title):
    """Generic function to perform linear regression for any metric vs client change fraction."""
    conv_ids = sorted(df['conv_id'].unique())
    
    change_fractions = []
    metric_values = []
    valid_conv_ids = []
    
    # Calculate metrics for each conversation
    for conv_id in conv_ids:
        change_frac = calculate_client_change_fraction(conv_id)
        metric_val = metric_func(df, conv_id)
        
        if change_frac is not None and metric_val is not None:
            change_fractions.append(change_frac)
            metric_values.append(metric_val)
            valid_conv_ids.append(conv_id)
    
    if len(change_fractions) < 2:
        print(f"Not enough valid data points for regression analysis of {metric_name}")
        return
    
    # Convert to numpy arrays for regression
    X = np.array(change_fractions).reshape(-1, 1)
    y = np.array(metric_values)
    
    # Perform linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate statistics
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, y_pred)
    
    # Calculate correlation and p-value
    correlation, p_value = stats.pearsonr(change_fractions, metric_values)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(change_fractions, metric_values, alpha=0.6, s=100, label='Conversations')
    
    # Regression line
    plt.plot(change_fractions, y_pred, color='red', linewidth=2, label=f'Regression line')
    
    # Add conversation IDs as labels
    for i, conv_id in enumerate(valid_conv_ids):
        plt.annotate(f'{conv_id}', (change_fractions[i], metric_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Client Change Fraction (C/(C+S))', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(plot_title, fontsize=14)
    
    # Add statistics to plot
    stats_text = f'y = {slope:.3f}x + {intercept:.3f}\n'
    stats_text += f'R² = {r2:.3f}\n'
    stats_text += f'Correlation = {correlation:.3f}\n'
    stats_text += f'p-value = {p_value:.4f}'
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print(f"\n{plot_title}")
    print("=" * 50)
    print(f"Number of conversations analyzed: {len(valid_conv_ids)}")
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R-squared: {r2:.4f}")
    print(f"Pearson correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Print data table
    print("\nDetailed Data:")
    print("-" * 50)
    print(f"{'Conv ID':>8} | {'Change Fraction':>15} | {metric_name:>20}")
    print("-" * 50)
    for i, conv_id in enumerate(valid_conv_ids):
        print(f"{conv_id:>8} | {change_fractions[i]:>15.3f} | {metric_values[i]:>20.2f}")


def main():
    df = read_and_preprocess_csv("/Users/joeberson/Developer/AutoMISC/data/annotated/random_install_multiBC_20convos_all_tiered_gpt-4o_interval_5_annotated.csv")
    
    # Original analysis - Average BCs per turn
    print("\n1. Performing average BCs per turn regression analysis...")
    perform_regression_analysis(
        df, 
        calculate_average_bcs_per_turn,
        "Average BCs per Counsellor Turn",
        "Linear Regression: Average BCs per Turn vs Client Change Fraction"
    )
    
    # New analysis 1 - Single BC volleys
    print("\n2. Performing single BC volleys regression analysis...")
    perform_regression_analysis(
        df,
        count_single_bc_volleys,
        "Count of Single BC Volleys",
        "Linear Regression: Single BC Volleys vs Client Change Fraction"
    )
    
    # New analysis 2 - Double BC volleys
    print("\n3. Performing double BC volleys regression analysis...")
    perform_regression_analysis(
        df,
        count_double_bc_volleys,
        "Count of Two BC Volleys",
        "Linear Regression: Two BC Volleys vs Client Change Fraction"
    )


if __name__ == "__main__":
    main()