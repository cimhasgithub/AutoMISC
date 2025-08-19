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


def filter_last_volley_rows(df):
    """Filters and returns only the last volley row per Volley #."""
    cols = [
        "conv_id", "speaker", "corp_vol_idx", "vol_text", "BC Pattern T1", "BC Pattern T2"
    ]
    df_filtered = df[cols]
    return df_filtered.groupby("corp_vol_idx", as_index=False).last()


def plot_bc_pattern_distribution(df, pattern_col, title, threshold):
    """Plots pie chart for BC pattern distribution based on frequency threshold."""
    counts = df[pattern_col].value_counts(normalize=True)
    main = counts[counts >= threshold]
    other = counts[counts < threshold]
    combined = main.copy()
    combined["Other"] = other.sum()

    plt.figure(figsize=(8, 8))
    plt.pie(combined.values, labels=combined.index, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def get_bc_combination_counts_per_conversation(df, pattern_col="BC Pattern T1", speaker="counsellor"):
    """
    Count occurrences of each BC combination per conversation.
    Returns a dictionary with conversation IDs as keys and dictionaries of BC pattern counts as values.
    """
    # Filter for specific speaker
    speaker_df = df[df["speaker"].str.lower() == speaker]
    
    # Group by conversation and count BC patterns
    conv_bc_counts = {}
    for conv_id in speaker_df["conv_id"].unique():
        conv_data = speaker_df[speaker_df["conv_id"] == conv_id]
        bc_counts = conv_data[pattern_col].value_counts().to_dict()
        conv_bc_counts[conv_id] = bc_counts
    
    return conv_bc_counts


def get_delta_confidence_per_conversation(df_metadata):
    """
    Get the delta confidence value for each conversation from metadata.
    Maps Participant id from df_metadata to conv_id.
    """
    delta_conf = {}
    for _, row in df_metadata.iterrows():
        participant_id = row["Participant id"]
        delta_confidence = row["delta_confidence"]
        if pd.notna(delta_confidence):
            delta_conf[participant_id] = delta_confidence
    
    return delta_conf


def get_care_ratings_per_conversation(df_care_ratings):
    """
    Get the average care rating value for each conversation from care ratings data.
    Maps id from df_care_ratings to conv_id.
    """
    care_ratings = {}
    for _, row in df_care_ratings.iterrows():
        conv_id = row["id"]
        avg_care_rating = row["average_care_rating"]
        if pd.notna(avg_care_rating):
            care_ratings[conv_id] = avg_care_rating
    
    return care_ratings


def perform_bc_combination_regression(df, df_metadata, bc_combination, pattern_col="BC Pattern T1", speaker="counsellor"):
    """
    Perform linear regression for a specific BC combination.
    X: Number of times the BC combination appears in a conversation
    Y: Delta confidence of the conversation
    """
    # Get BC combination counts per conversation
    conv_bc_counts = get_bc_combination_counts_per_conversation(df, pattern_col, speaker)
    
    # Get delta confidence per conversation
    delta_conf = get_delta_confidence_per_conversation(df_metadata)
    
    # Prepare data for regression
    conv_ids = []
    x_values = []  # Count of specific BC combination
    y_values = []  # Delta confidence
    
    for conv_id in conv_bc_counts.keys():
        if conv_id in delta_conf:
            # Get count of specific BC combination (0 if not present)
            bc_count = conv_bc_counts[conv_id].get(bc_combination, 0)
            x_values.append(bc_count)
            y_values.append(delta_conf[conv_id])
            conv_ids.append(conv_id)
    
    if len(x_values) < 2:
        print(f"Not enough data points for regression analysis of '{bc_combination}'")
        return None
    
    # Convert to numpy arrays
    X = np.array(x_values).reshape(-1, 1)
    y = np.array(y_values)
    
    # Perform linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate statistics
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, y_pred)
    correlation, p_value = stats.pearsonr(x_values, y_values)
    
    return {
        'bc_combination': bc_combination,
        'slope': slope,
        'intercept': intercept,
        'r2': r2,
        'correlation': correlation,
        'p_value': p_value,
        'x_values': x_values,
        'y_values': y_values,
        'y_pred': y_pred,
        'conv_ids': conv_ids
    }


def perform_bc_combination_care_rating_regression(df, df_care_ratings, bc_combination, pattern_col="BC Pattern T1", speaker="counsellor"):
    """
    Perform linear regression for a specific BC combination against care ratings.
    X: Number of times the BC combination appears in a conversation
    Y: Average care rating of the conversation
    """
    # Get BC combination counts per conversation
    conv_bc_counts = get_bc_combination_counts_per_conversation(df, pattern_col, speaker)
    
    # Get care ratings per conversation
    care_ratings = get_care_ratings_per_conversation(df_care_ratings)
    
    # Prepare data for regression
    conv_ids = []
    x_values = []  # Count of specific BC combination
    y_values = []  # Average care rating
    
    for conv_id in conv_bc_counts.keys():
        if conv_id in care_ratings:
            # Get count of specific BC combination (0 if not present)
            bc_count = conv_bc_counts[conv_id].get(bc_combination, 0)
            x_values.append(bc_count)
            y_values.append(care_ratings[conv_id])
            conv_ids.append(conv_id)
    
    if len(x_values) < 2:
        print(f"Not enough data points for care rating regression analysis of '{bc_combination}'")
        return None
    
    # Convert to numpy arrays
    X = np.array(x_values).reshape(-1, 1)
    y = np.array(y_values)
    
    # Perform linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate statistics
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, y_pred)
    correlation, p_value = stats.pearsonr(x_values, y_values)
    
    return {
        'bc_combination': bc_combination,
        'slope': slope,
        'intercept': intercept,
        'r2': r2,
        'correlation': correlation,
        'p_value': p_value,
        'x_values': x_values,
        'y_values': y_values,
        'y_pred': y_pred,
        'conv_ids': conv_ids
    }


def plot_bc_regression(result, save_path=None, y_label='Delta Confidence', title_suffix='vs Delta Confidence'):
    """Plot the regression results for a BC combination."""
    if result is None:
        return
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(result['x_values'], result['y_values'], alpha=0.6, s=100, label='Conversations')
    
    # Regression line
    plt.plot(result['x_values'], result['y_pred'], color='red', linewidth=2, label='Regression line')
    
    # Conversation IDs removed for cleaner visualization
    
    plt.xlabel(f"Count of '{result['bc_combination']}' BC Combination", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(f"Linear Regression: '{result['bc_combination']}' {title_suffix}", fontsize=14)
    
    # Add statistics to plot
    stats_text = f'y = {result["slope"]:.4f}x + {result["intercept"]:.4f}\n'
    stats_text += f'R² = {result["r2"]:.4f}\n'
    stats_text += f'Correlation = {result["correlation"]:.4f}\n'
    stats_text += f'p-value = {result["p_value"]:.4f}'
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def analyze_all_bc_combinations(df, df_metadata, pattern_col="BC Pattern T1", speaker="counsellor", min_occurrences=5):
    """
    Analyze all BC combinations that appear at least min_occurrences times across all conversations.
    Returns a sorted list of results by absolute correlation.
    """
    # Get all BC combinations and their total counts
    speaker_df = df[df["speaker"].str.lower() == speaker]
    bc_counts_total = speaker_df[pattern_col].value_counts()
    
    # Filter by minimum occurrences
    frequent_bcs = bc_counts_total[bc_counts_total >= min_occurrences].index.tolist()
    
    results = []
    for bc_combination in frequent_bcs:
        result = perform_bc_combination_regression(df, df_metadata, bc_combination, pattern_col, speaker)
        if result:
            results.append(result)
    
    # Sort by absolute correlation
    results.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return results


def analyze_all_bc_combinations_care_rating(df, df_care_ratings, pattern_col="BC Pattern T1", speaker="counsellor", min_occurrences=5):
    """
    Analyze all BC combinations against care ratings that appear at least min_occurrences times across all conversations.
    Returns a sorted list of results by absolute correlation.
    """
    # Get all BC combinations and their total counts
    speaker_df = df[df["speaker"].str.lower() == speaker]
    bc_counts_total = speaker_df[pattern_col].value_counts()
    
    # Filter by minimum occurrences
    frequent_bcs = bc_counts_total[bc_counts_total >= min_occurrences].index.tolist()
    
    results = []
    for bc_combination in frequent_bcs:
        result = perform_bc_combination_care_rating_regression(df, df_care_ratings, bc_combination, pattern_col, speaker)
        if result:
            results.append(result)
    
    # Sort by absolute correlation
    results.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return results


def print_regression_summary(results, top_n=10, title="Linear Regression Analysis Summary"):
    """Print a summary table of regression results."""
    print(f"\n{title}:")
    print("=" * 100)
    print(f"{'BC Combination':30} | {'Slope':>10} | {'R²':>8} | {'Corr':>8} | {'p-value':>10} | {'Significant':>12}")
    print("-" * 100)
    
    for i, result in enumerate(results[:top_n]):
        sig = "Yes" if result['p_value'] < 0.05 else "No"
        bc_comb = result['bc_combination'][:28] + ".." if len(result['bc_combination']) > 30 else result['bc_combination']
        print(f"{bc_comb:30} | {result['slope']:>10.4f} | {result['r2']:>8.4f} | "
              f"{result['correlation']:>8.4f} | {result['p_value']:>10.4f} | {sig:>12}")
    
    if len(results) > top_n:
        print(f"\n... and {len(results) - top_n} more BC combinations")


def analyze_total_bc_diversity(df, df_metadata, pattern_col="BC Pattern T1", speaker="counsellor"):
    """
    Analyze the relationship between total number of unique BC combinations used
    in a conversation and delta confidence.
    """
    speaker_df = df[df["speaker"].str.lower() == speaker]
    delta_conf = get_delta_confidence_per_conversation(df_metadata)
    
    conv_ids = []
    x_values = []  # Number of unique BC combinations
    y_values = []  # Delta confidence
    
    for conv_id in speaker_df["conv_id"].unique():
        if conv_id in delta_conf:
            conv_data = speaker_df[speaker_df["conv_id"] == conv_id]
            # Count unique BC combinations in this conversation
            unique_bcs = conv_data[pattern_col].nunique()
            x_values.append(unique_bcs)
            y_values.append(delta_conf[conv_id])
            conv_ids.append(conv_id)
    
    if len(x_values) < 2:
        print("Not enough data points for diversity analysis")
        return None
    
    # Perform regression
    X = np.array(x_values).reshape(-1, 1)
    y = np.array(y_values)
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, y_pred)
    correlation, p_value = stats.pearsonr(x_values, y_values)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(x_values, y_values, alpha=0.6, s=100, label='Conversations')
    plt.plot(x_values, y_pred, color='red', linewidth=2, label='Regression line')
    
    # Conversation IDs removed for cleaner visualization
    
    plt.xlabel('Number of Unique BC Combinations Used', fontsize=12)
    plt.ylabel('Delta Confidence', fontsize=12)
    plt.title('Linear Regression: BC Diversity vs Delta Confidence', fontsize=14)
    
    stats_text = f'y = {slope:.4f}x + {intercept:.4f}\n'
    stats_text += f'R² = {r2:.4f}\n'
    stats_text += f'Correlation = {correlation:.4f}\n'
    stats_text += f'p-value = {p_value:.4f}'
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("\nBC Diversity Analysis:")
    print("=" * 50)
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R-squared: {r2:.4f}")
    print(f"Correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")


def main():
    # Load and preprocess data
    df = read_and_preprocess_csv("/Users/joeberson/Developer/AutoMISC/data/MIV6.3A_lowconf_tiered_gpt-4.1-2025-04-14_interval_3_annotated_copy.csv")
    df_metadata = pd.read_csv("/Users/joeberson/Developer/AutoMISC/data/2024-11-14-MIV6.3A-2024-11-22-MIV6.3A_all_data_delta_with_post_keep_high_conf_True_merged.csv")
    df_care_ratings = pd.read_csv("/Users/joeberson/Developer/AutoMISC/data/care_ratings.csv")
    df_last = filter_last_volley_rows(df)
    
    # Analyze specific BC combinations
    print("\nAnalyzing BC combinations vs Delta Confidence...")
    print("=" * 50)
    
    # Option 1: Analyze all frequent BC combinations
    results = analyze_all_bc_combinations(df_last, df_metadata, pattern_col="BC Pattern T2", 
                                         speaker="counsellor", min_occurrences=5)
    
    # Print summary of top correlations
    print_regression_summary(results, top_n=10)
    
    # Plot top 3 most correlated BC combinations
    print("\nPlotting top 3 most correlated BC combinations...")
    for i, result in enumerate(results[:3]):
        plot_bc_regression(result)
    
    # # Option 2: Analyze specific BC combinations of interest
    # specific_bcs = ["R", "S", "R, S", "Q, R", "Q, R, S"]  # Modify as needed
    # print("\n\nAnalyzing specific BC combinations:")
    # print("=" * 50)
    
    # for bc_combo in specific_bcs:
    #     result = perform_bc_combination_regression(df_last, df_metadata, bc_combo, 
    #                                               pattern_col="BC Pattern T2", 
    #                                               speaker="counsellor")
    #     if result:
    #         print(f"\n{bc_combo}:")
    #         print(f"  Correlation: {result['correlation']:.4f}")
    #         print(f"  P-value: {result['p_value']:.4f}")
    #         print(f"  R²: {result['r2']:.4f}")
    #         # Optionally plot
    #         # plot_bc_regression(result)
    
    # Option 3: Analyze BC diversity (number of unique combinations used)
    print("\n\nAnalyzing BC diversity vs Delta Confidence...")
    analyze_total_bc_diversity(df_last, df_metadata, pattern_col="BC Pattern T2", speaker="counsellor")
    
    # Option 4: Analyze all frequent BC combinations vs Care Ratings
    print("\n\nAnalyzing BC combinations vs Average Care Rating...")
    print("=" * 50)
    
    care_rating_results = analyze_all_bc_combinations_care_rating(df_last, df_care_ratings, 
                                                                pattern_col="BC Pattern T2", 
                                                                speaker="counsellor", min_occurrences=5)
    
    # Print summary of top correlations for care ratings
    print_regression_summary(care_rating_results, top_n=10, title="BC Combinations vs Average Care Rating Analysis Summary")
    
    # Plot top 3 most correlated BC combinations for care ratings
    print("\nPlotting top 3 most correlated BC combinations vs Care Rating...")
    for i, result in enumerate(care_rating_results[:3]):
        plot_bc_regression(result, y_label='Average Care Rating', title_suffix='vs Average Care Rating')


if __name__ == "__main__":
    main()