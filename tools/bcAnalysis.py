import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import textwrap


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


def get_bc_combination_counts_per_conversation(df, pattern_col="BC Pattern T1", speaker="counsellor"):
    """
    Count occurrences of each BC combination per conversation.
    Returns a dictionary with conversation IDs as keys and dictionaries of BC pattern counts as values.
    """
    speaker_df = df[df["speaker"].str.lower() == speaker]
    
    conv_bc_counts = {}
    for conv_id in speaker_df["conv_id"].unique():
        conv_data = speaker_df[speaker_df["conv_id"] == conv_id]
        bc_counts = conv_data[pattern_col].value_counts().to_dict()
        conv_bc_counts[conv_id] = bc_counts
    
    return conv_bc_counts


def get_delta_confidence_per_conversation(df_metadata):
    """Get the delta confidence value for each conversation from metadata."""
    delta_conf = {}
    for _, row in df_metadata.iterrows():
        participant_id = row["Participant id"]
        delta_confidence = row["delta_confidence"]
        if pd.notna(delta_confidence):
            delta_conf[participant_id] = delta_confidence
    
    return delta_conf


def get_care_ratings_per_conversation(df_care_ratings):
    """Get the average care rating value for each conversation from care ratings data."""
    care_ratings = {}
    for _, row in df_care_ratings.iterrows():
        conv_id = row["id"]
        avg_care_rating = row["average_care_rating"]
        if pd.notna(avg_care_rating):
            care_ratings[conv_id] = avg_care_rating
    
    return care_ratings


def perform_bc_combination_regression(df, df_metadata, bc_combination, pattern_col="BC Pattern T1", speaker="counsellor"):
    """Perform linear regression for a specific BC combination."""
    conv_bc_counts = get_bc_combination_counts_per_conversation(df, pattern_col, speaker)
    delta_conf = get_delta_confidence_per_conversation(df_metadata)
    
    conv_ids = []
    x_values = []
    y_values = []
    
    for conv_id in conv_bc_counts.keys():
        if conv_id in delta_conf:
            bc_count = conv_bc_counts[conv_id].get(bc_combination, 0)
            x_values.append(bc_count)
            y_values.append(delta_conf[conv_id])
            conv_ids.append(conv_id)
    
    if len(x_values) < 2:
        return None
    
    X = np.array(x_values).reshape(-1, 1)
    y = np.array(y_values)
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
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


def analyze_all_bc_combinations(df, df_metadata, pattern_col="BC Pattern T1", speaker="counsellor", min_occurrences=5):
    """Analyze all BC combinations that appear at least min_occurrences times."""
    speaker_df = df[df["speaker"].str.lower() == speaker]
    bc_counts_total = speaker_df[pattern_col].value_counts()
    
    frequent_bcs = bc_counts_total[bc_counts_total >= min_occurrences].index.tolist()
    
    results = []
    for bc_combination in frequent_bcs:
        result = perform_bc_combination_regression(df, df_metadata, bc_combination, pattern_col, speaker)
        if result:
            results.append(result)
    
    results.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return results


def extract_volleys_for_bc_combination(df_full, df_last, bc_combination, pattern_col="BC Pattern T2", 
                                      speaker="counsellor", max_volleys=None):
    """
    Extract actual volley texts for a specific BC combination.
    Returns a DataFrame with conversation details and volley texts.
    
    Parameters:
    - max_volleys: Maximum number of volleys to return (None = return all)
    """
    # Filter for the specific BC combination and speaker
    mask = (df_last[pattern_col] == bc_combination) & (df_last["speaker"].str.lower() == speaker)
    matching_volleys = df_last[mask].copy()
    
    if matching_volleys.empty:
        return pd.DataFrame()
    
    # Get full volley text from original dataframe
    volley_details = []
    
    for _, row in matching_volleys.iterrows():
        corp_vol_idx = row['corp_vol_idx']
        conv_id = row['conv_id']
        
        # Get all utterances in this volley from the full dataframe
        volley_utterances = df_full[df_full['corp_vol_idx'] == corp_vol_idx].sort_values('corp_utt_idx')
        
        # Combine all utterances in the volley
        full_volley_text = ' '.join(volley_utterances['utt_text'].fillna('').astype(str))
        
        volley_details.append({
            'conv_id': conv_id,
            'corp_vol_idx': corp_vol_idx,
            'speaker': row['speaker'],
            'bc_pattern': bc_combination,
            'volley_text_summary': row['vol_text'][:200] + '...' if len(str(row['vol_text'])) > 200 else row['vol_text'],
            'full_volley_text': full_volley_text,  # No truncation for CSV
            'display_volley_text': full_volley_text[:500] + '...' if len(full_volley_text) > 500 else full_volley_text  # Truncated version for display
        })
    
    result_df = pd.DataFrame(volley_details)
    
    # Limit to max_volleys if specified
    if max_volleys and len(result_df) > max_volleys:
        result_df = result_df.sample(n=max_volleys, random_state=42)
    
    return result_df


def analyze_significant_bc_volleys(df_full, df_last, df_metadata, pattern_col="BC Pattern T2", 
                                  speaker="counsellor", top_n=5, volleys_per_bc=5):
    """
    Analyze the most significant BC combinations and extract example volleys.
    """
    # Get regression results for all BC combinations
    results = analyze_all_bc_combinations(df_last, df_metadata, pattern_col=pattern_col, 
                                         speaker=speaker, min_occurrences=5)
    
    if not results:
        print("No significant BC combinations found.")
        return None
    
    # Create summary report
    print("=" * 100)
    print(f"TOP {top_n} MOST SIGNIFICANT BC COMBINATIONS")
    print("=" * 100)
    
    all_volley_data = []
    
    for i, result in enumerate(results[:top_n], 1):
        bc_combo = result['bc_combination']
        
        print(f"\n{i}. BC Combination: '{bc_combo}'")
        print("-" * 80)
        print(f"   Correlation: {result['correlation']:.4f}")
        print(f"   P-value: {result['p_value']:.4f}")
        print(f"   R²: {result['r2']:.4f}")
        print(f"   Slope: {result['slope']:.4f}")
        
        # Count conversations with this BC combination (count > 0)
        convs_with_bc = sum(1 for x in result['x_values'] if x > 0)
        total_occurrences = sum(result['x_values'])
        
        print(f"   Conversations with this BC: {convs_with_bc}/{len(result['x_values'])}")
        print(f"   Total occurrences: {total_occurrences}")
        
        # Extract example volleys
        volley_df = extract_volleys_for_bc_combination(df_full, df_last, bc_combo, 
                                                       pattern_col=pattern_col, 
                                                       speaker=speaker, 
                                                       max_volleys=volleys_per_bc)
        
        if not volley_df.empty:
            print(f"\n   EXAMPLE VOLLEYS (showing up to {volleys_per_bc}):")
            print("   " + "=" * 75)
            
            for j, (_, volley_row) in enumerate(volley_df.iterrows(), 1):
                print(f"\n   Example {j}:")
                print(f"   Conversation ID: {volley_row['conv_id']}")
                print(f"   Volley Index: {volley_row['corp_vol_idx']}")
                # Use truncated version for display
                display_text = volley_row.get('display_volley_text', volley_row['full_volley_text'][:500] + '...')
                print(f"   Text: {textwrap.fill(display_text, width=90, initial_indent='   ', subsequent_indent='        ')}")
                print("   " + "-" * 70)
            
            # Add to overall data
            volley_df['significance_rank'] = i
            volley_df['correlation'] = result['correlation']
            volley_df['p_value'] = result['p_value']
            all_volley_data.append(volley_df)
        else:
            print(f"\n   No volleys found for this BC combination.")
    
    # Combine all volley data
    if all_volley_data:
        combined_volley_df = pd.concat(all_volley_data, ignore_index=True)
        return combined_volley_df
    
    return None


def create_detailed_volley_report(df_full, df_last, df_metadata, pattern_col="BC Pattern T2", 
                                 speaker="counsellor", output_file=None, top_n=5, volleys_per_bc=5):
    """
    Create a detailed report of significant BC combinations with their volleys.
    
    Parameters:
    - top_n: Number of top BC combinations to analyze
    - volleys_per_bc: Max volleys to show per BC (None = show all)
    """
    combined_df = analyze_significant_bc_volleys(df_full, df_last, df_metadata, 
                                                pattern_col=pattern_col, 
                                                speaker=speaker, 
                                                top_n=top_n, 
                                                volleys_per_bc=volleys_per_bc)
    
    if combined_df is not None and output_file:
        # Save to CSV for further analysis
        combined_df.to_csv(output_file, index=False)
        print(f"\n\nDetailed volley data saved to: {output_file}")
    
    return combined_df


def find_volleys_for_conversations_with_bc(df_full, df_last, result, bc_combination, 
                                          pattern_col="BC Pattern T2", speaker="counsellor"):
    """
    For a given BC combination regression result, find the actual volleys 
    in conversations where the BC appears (count > 0).
    """
    # Get conversations where this BC combination appears
    convs_with_bc = [(conv_id, count) for conv_id, count in 
                     zip(result['conv_ids'], result['x_values']) if count > 0]
    
    if not convs_with_bc:
        print(f"No conversations found with BC combination '{bc_combination}'")
        return None
    
    print(f"\nConversations with BC combination '{bc_combination}':")
    print("=" * 80)
    
    volley_data = []
    
    for conv_id, count in convs_with_bc:
        print(f"\nConversation: {conv_id} (appears {count} time(s))")
        print("-" * 60)
        
        # Find the volleys in this conversation with this BC
        mask = (df_last['conv_id'] == conv_id) & \
               (df_last[pattern_col] == bc_combination) & \
               (df_last['speaker'].str.lower() == speaker)
        
        matching_volleys = df_last[mask]
        
        for _, volley in matching_volleys.iterrows():
            corp_vol_idx = volley['corp_vol_idx']
            
            # Get full volley from original dataframe
            volley_utterances = df_full[df_full['corp_vol_idx'] == corp_vol_idx].sort_values('corp_utt_idx')
            full_text = ' '.join(volley_utterances['utt_text'].fillna('').astype(str))
            
            print(f"  Volley {corp_vol_idx}:")
            wrapped_text = textwrap.fill(full_text[:500], width=70, 
                                        initial_indent='    ', 
                                        subsequent_indent='    ')
            print(wrapped_text)
            if len(full_text) > 500:
                print("    ...")
            print()
            
            volley_data.append({
                'conv_id': conv_id,
                'corp_vol_idx': corp_vol_idx,
                'bc_combination': bc_combination,
                'occurrence_count': count,
                'full_text': full_text
            })
    
    return pd.DataFrame(volley_data)


def plot_bc_regression(result, save_path=None, y_label='Delta Confidence', title_suffix='vs Delta Confidence'):
    """Plot the regression results for a BC combination."""
    if result is None:
        return
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(result['x_values'], result['y_values'], alpha=0.6, s=100, label='Conversations')
    
    # Regression line
    plt.plot(result['x_values'], result['y_pred'], color='red', linewidth=2, label='Regression line')
    
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


# Main execution function
def main():
    # File paths - update these to your actual paths
    csv_path = "/Users/joeberson/Developer/AutoMISC/data/MIV6.3A_lowconf_tiered_gpt-4.1-2025-04-14_interval_3_annotated_copy.csv"
    metadata_path = "/Users/joeberson/Developer/AutoMISC/data/2024-11-14-MIV6.3A-2024-11-22-MIV6.3A_all_data_delta_with_post_keep_high_conf_True_merged.csv"
    care_ratings_path = "/Users/joeberson/Developer/AutoMISC/data/care_ratings.csv"
    
    # Load data
    print("Loading data...")
    df_full = read_and_preprocess_csv(csv_path)
    df_metadata = pd.read_csv(metadata_path)
    df_care_ratings = pd.read_csv(care_ratings_path)
    df_last = filter_last_volley_rows(df_full)
    
    # Create detailed report with volleys
    print("\nAnalyzing most significant BC combinations and extracting volleys...")
    # Adjust these parameters as needed:
    # - top_n: number of BC combinations to analyze (e.g., 10 for top 10)
    # - volleys_per_bc: max volleys to show per BC (e.g., None for all, or 20 for more examples)
    volley_report = create_detailed_volley_report(
        df_full, df_last, df_metadata,
        pattern_col="BC Pattern T2",
        speaker="counsellor",
        output_file="significant_bc_volleys.csv",
        top_n=10,  # Analyze top 10 BC combinations
        volleys_per_bc=None  # Show ALL volleys (no limit)
    )
    
    # Additional analysis: Find volleys for specific BC combinations
    print("\n" + "=" * 100)
    print("DETAILED ANALYSIS OF SPECIFIC BC COMBINATIONS")
    print("=" * 100)
    
    # Get the most significant result
    results = analyze_all_bc_combinations(df_last, df_metadata, 
                                         pattern_col="BC Pattern T2", 
                                         speaker="counsellor", 
                                         min_occurrences=5)
    
    if results:
        most_significant = results[0]
        bc_to_analyze = most_significant['bc_combination']
        
        print(f"\nMost significant BC combination: '{bc_to_analyze}'")
        volley_details = find_volleys_for_conversations_with_bc(
            df_full, df_last, most_significant, bc_to_analyze,
            pattern_col="BC Pattern T2", speaker="counsellor"
        )
        
        if volley_details is not None:
            print(f"\nFound {len(volley_details)} volleys with this BC combination")
            
    # Print summary of all results
    print("\n" + "=" * 100)
    print_regression_summary(results, top_n=10)
    
    # Optional: Plot top 3 most correlated BC combinations
    print("\nGenerating plots for top 3 most correlated BC combinations...")
    for i, result in enumerate(results[:3]):
        plot_bc_regression(result, save_path=f"bc_regression_{i+1}.png")
    
    # Optional: Analyze BC diversity
    print("\n" + "=" * 100)
    print("ANALYZING BC DIVERSITY")
    print("=" * 100)
    analyze_total_bc_diversity(df_last, df_metadata, pattern_col="BC Pattern T2", speaker="counsellor")
    
    return volley_report


if __name__ == "__main__":
    volley_data = main()