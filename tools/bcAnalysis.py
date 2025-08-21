import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import textwrap
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split


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
                                  speaker="counsellor", top_n=5, volleys_per_bc=5, 
                                  print_volleys=False):
    """
    Analyze the most significant BC combinations and extract example volleys.
    Set print_volleys=False to suppress terminal output.
    """
    # Get regression results for all BC combinations
    results = analyze_all_bc_combinations_corrected(df_last, df_metadata, pattern_col=pattern_col, 
                                                   speaker=speaker, min_occurrences=5)
    
    if not results:
        print("No significant BC combinations found.")
        return None
    
    # Create summary report (keep this - it's concise)
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
        if 'p_adjusted' in result:
            print(f"   Adjusted P-value: {result['p_adjusted']:.4f}")
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
            if print_volleys:
                # Only print volleys if explicitly requested
                print(f"\n   EXAMPLE VOLLEYS (showing up to {volleys_per_bc}):")
                print("   " + "=" * 75)
                
                for j, (_, volley_row) in enumerate(volley_df.iterrows(), 1):
                    print(f"\n   Example {j}:")
                    print(f"   Conversation ID: {volley_row['conv_id']}")
                    print(f"   Volley Index: {volley_row['corp_vol_idx']}")
                    display_text = volley_row.get('display_volley_text', volley_row['full_volley_text'][:500] + '...')
                    print(f"   Text: {textwrap.fill(display_text, width=90, initial_indent='   ', subsequent_indent='        ')}")
                    print("   " + "-" * 70)
            else:
                # Just mention that volleys were found and saved
                print(f"   → {len(volley_df)} example volleys found and saved to CSV")
            
            # Add to overall data
            volley_df['significance_rank'] = i
            volley_df['correlation'] = result['correlation']
            volley_df['p_value'] = result['p_value']
            if 'p_adjusted' in result:
                volley_df['p_adjusted'] = result['p_adjusted']
            all_volley_data.append(volley_df)
        else:
            print(f"   → No volleys found for this BC combination.")
    
    # Combine all volley data
    if all_volley_data:
        combined_volley_df = pd.concat(all_volley_data, ignore_index=True)
        return combined_volley_df
    
    return None

def create_detailed_volley_report(df_full, df_last, df_metadata, pattern_col="BC Pattern T2", 
                                 speaker="counsellor", output_file=None, top_n=5, 
                                 volleys_per_bc=5, print_volleys=False):
    """
    Create a detailed report of significant BC combinations with their volleys.
    Set print_volleys=False to suppress volley terminal output (DEFAULT).
    """
    combined_df = analyze_significant_bc_volleys(df_full, df_last, df_metadata, 
                                                pattern_col=pattern_col, 
                                                speaker=speaker, 
                                                top_n=top_n, 
                                                volleys_per_bc=volleys_per_bc,
                                                print_volleys=print_volleys)
    
    if combined_df is not None and output_file:
        # Save to CSV for further analysis
        combined_df.to_csv(output_file, index=False)
        print(f"\n💾 Detailed volley data saved to: {output_file}")
        print(f"   Total volleys saved: {len(combined_df)}")
        print(f"   Columns: {', '.join(combined_df.columns.tolist())}")
    
    return combined_df


def find_volleys_for_conversations_with_bc(df_full, df_last, result, bc_combination, 
                                          pattern_col="BC Pattern T2", speaker="counsellor",
                                          print_volleys=False):
    """
    For a given BC combination regression result, find the actual volleys 
    in conversations where the BC appears (count > 0).
    Set print_volleys=False to suppress terminal output.
    """
    # Get conversations where this BC combination appears
    convs_with_bc = [(conv_id, count) for conv_id, count in 
                     zip(result['conv_ids'], result['x_values']) if count > 0]
    
    if not convs_with_bc:
        print(f"No conversations found with BC combination '{bc_combination}'")
        return None
    
    if print_volleys:
        print(f"\nConversations with BC combination '{bc_combination}':")
        print("=" * 80)
    else:
        print(f"Found {len(convs_with_bc)} conversations with BC combination '{bc_combination}'")
    
    volley_data = []
    
    for conv_id, count in convs_with_bc:
        if print_volleys:
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
            
            if print_volleys:
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

def analyze_all_bc_combinations_corrected(df, df_metadata, pattern_col="BC Pattern T1", 
                                        speaker="counsellor", min_occurrences=5, 
                                        correction_method='fdr_bh'):
    """
    REPLACES your existing analyze_all_bc_combinations function.
    Analyze all BC combinations with proper multiple testing correction.
    """
    speaker_df = df[df["speaker"].str.lower() == speaker]
    bc_counts_total = speaker_df[pattern_col].value_counts()
    
    frequent_bcs = bc_counts_total[bc_counts_total >= min_occurrences].index.tolist()
    
    results = []
    for bc_combination in frequent_bcs:
        result = perform_bc_combination_regression(df, df_metadata, bc_combination, pattern_col, speaker)
        if result:
            results.append(result)
    
    # Apply multiple testing correction
    if results and correction_method:
        p_values = [result['p_value'] for result in results]
        
        if correction_method == 'fdr_bh':
            rejected, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
            correction_name = "FDR (Benjamini-Hochberg)"
        elif correction_method == 'bonferroni':
            rejected, p_adjusted, _, _ = multipletests(p_values, method='bonferroni')
            correction_name = "Bonferroni"
        else:
            p_adjusted = p_values
            rejected = [p < 0.05 for p in p_values]
            correction_name = "None"
        
        # Add corrected values to results
        for i, result in enumerate(results):
            result['p_adjusted'] = p_adjusted[i]
            result['significant_raw'] = result['p_value'] < 0.05
            result['significant_corrected'] = rejected[i]
            result['correction_method'] = correction_name
        
        print(f"\nMultiple Testing Correction Applied: {correction_name}")
        print(f"Total tests performed: {len(results)}")
        print(f"Significant before correction: {sum(result['significant_raw'] for result in results)}")
        print(f"Significant after correction: {sum(result['significant_corrected'] for result in results)}")
    
    # Sort by absolute correlation (effect size) rather than p-value
    results.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return results


def print_corrected_summary(results, top_n=10):
    """REPLACES your existing print_regression_summary function."""
    print(f"\nRegression Analysis Summary (Top {top_n} by Effect Size):")
    print("=" * 120)
    print(f"{'BC Combination':30} | {'Corr':>8} | {'Raw p':>10} | {'Adj p':>10} | {'Raw Sig':>8} | {'Adj Sig':>8} | {'Effect':>8}")
    print("-" * 120)
    
    for i, result in enumerate(results[:top_n]):
        raw_sig = "Yes" if result.get('significant_raw', result['p_value'] < 0.05) else "No"
        adj_sig = "Yes" if result.get('significant_corrected', False) else "No"
        
        # Effect size interpretation
        abs_corr = abs(result['correlation'])
        if abs_corr >= 0.5:
            effect = "Large"
        elif abs_corr >= 0.3:
            effect = "Medium"
        else:
            effect = "Small"
        
        bc_comb = result['bc_combination'][:28] + ".." if len(result['bc_combination']) > 30 else result['bc_combination']
        
        raw_p = result['p_value']
        adj_p = result.get('p_adjusted', 'N/A')
        
        adj_p_str = f"{adj_p:.4f}" if isinstance(adj_p, float) else str(adj_p)
        
        print(f"{bc_comb:30} | {result['correlation']:>8.3f} | {raw_p:>10.4f} | "
              f"{adj_p_str:>10} | {raw_sig:>8} | {adj_sig:>8} | {effect:>8}")


def effect_size_analysis(results, min_effect_size=0.3):
    """NEW FUNCTION - Add this to focus on practically significant effects."""
    print(f"\nEffect Size Analysis (|correlation| >= {min_effect_size}):")
    print("=" * 80)
    
    large_effects = [r for r in results if abs(r['correlation']) >= min_effect_size]
    
    if not large_effects:
        print(f"No BC combinations with correlation >= {min_effect_size}")
        return
    
    print(f"BC combinations with large effect sizes: {len(large_effects)}")
    
    for result in large_effects:
        direction = "positive" if result['correlation'] > 0 else "negative"
        print(f"\nBC: '{result['bc_combination']}'")
        print(f"  Correlation: {result['correlation']:.3f} ({direction})")
        print(f"  Raw p-value: {result['p_value']:.4f}")
        if 'p_adjusted' in result:
            print(f"  Adjusted p-value: {result['p_adjusted']:.4f}")
        print(f"  Effect size: {'Large' if abs(result['correlation']) >= 0.5 else 'Medium'}")


def train_test_validation(df, df_metadata, pattern_col="BC Pattern T1", 
                         speaker="counsellor", min_occurrences=5, test_size=0.5, random_state=42):
    """NEW FUNCTION - Add this for validation."""
    # Get unique conversation IDs
    conv_ids = df['conv_id'].unique()
    
    # Split conversations into train/test
    train_convs, test_convs = train_test_split(conv_ids, test_size=test_size, random_state=random_state)
    
    # Create train/test datasets
    df_train = df[df['conv_id'].isin(train_convs)]
    df_test = df[df['conv_id'].isin(test_convs)]
    
    print(f"Training conversations: {len(train_convs)}")
    print(f"Testing conversations: {len(test_convs)}")
    
    # Find significant BC combinations on training data
    train_results = analyze_all_bc_combinations_corrected(
        df_train, df_metadata, pattern_col, speaker, min_occurrences, 'fdr_bh'
    )
    
    # Test the top findings on test data
    significant_bcs = [r['bc_combination'] for r in train_results if r.get('significant_corrected', False)]
    
    if not significant_bcs:
        print("No significant BC combinations found in training data.")
        return None
    
    print(f"\nTesting {len(significant_bcs)} significant BC combinations on held-out test data:")
    
    test_results = []
    for bc_combination in significant_bcs:
        test_result = perform_bc_combination_regression(df_test, df_metadata, bc_combination, pattern_col, speaker)
        if test_result:
            # Find corresponding training result
            train_result = next((r for r in train_results if r['bc_combination'] == bc_combination), None)
            if train_result:
                test_result['train_correlation'] = train_result['correlation']
                test_result['train_p_value'] = train_result['p_value']
                test_result['train_significant'] = train_result.get('significant_corrected', False)
            test_results.append(test_result)
    
    return train_results, test_results

def compare_correction_methods(df, df_metadata, pattern_col="BC Pattern T2", 
                              speaker="counsellor", min_occurrences=5):
    """
    Compare FDR and Bonferroni corrections to assess robustness of findings.
    """
    # Run with FDR
    results_fdr = analyze_all_bc_combinations_corrected(
        df, df_metadata, pattern_col, speaker, min_occurrences, 
        correction_method='fdr_bh'
    )
    
    # Run with Bonferroni
    results_bonf = analyze_all_bc_combinations_corrected(
        df, df_metadata, pattern_col, speaker, min_occurrences, 
        correction_method='bonferroni'
    )
    
    # Create lookup dictionaries for easier comparison
    fdr_dict = {r['bc_combination']: r for r in results_fdr}
    bonf_dict = {r['bc_combination']: r for r in results_bonf}
    
    # Compare results
    print("\n" + "="*100)
    print("COMPARISON OF MULTIPLE TESTING CORRECTIONS")
    print("="*100)
    
    fdr_sig = [r['bc_combination'] for r in results_fdr if r['significant_corrected']]
    bonf_sig = [r['bc_combination'] for r in results_bonf if r['significant_corrected']]
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total BC combinations tested: {len(results_fdr)}")
    print(f"   Significant with FDR (α=0.05): {len(fdr_sig)} BC combinations")
    print(f"   Significant with Bonferroni (α=0.05): {len(bonf_sig)} BC combinations")
    print(f"   Bonferroni correction factor: {len(results_fdr)} (number of tests)")
    print(f"   Effective Bonferroni α threshold: {0.05/len(results_fdr):.6f}")
    
    # Find BC combinations that survive both corrections
    robust_findings = set(fdr_sig) & set(bonf_sig)
    fdr_only = set(fdr_sig) - set(bonf_sig)
    
    print(f"\n🔍 ROBUSTNESS ANALYSIS:")
    print(f"   Significant with BOTH corrections (most robust): {len(robust_findings)} BC combinations")
    print(f"   Significant with FDR only: {len(fdr_only)} BC combinations")
    
    # Detailed comparison table
    print("\n" + "="*100)
    print("DETAILED COMPARISON (Top 15 by effect size)")
    print("="*100)
    print(f"{'BC Combination':35} | {'Corr':>7} | {'Raw p':>10} | {'FDR p':>10} | {'Bonf p':>10} | {'FDR':>5} | {'Bonf':>5}")
    print("-"*100)
    
    for i, bc_combo in enumerate(fdr_dict.keys()):
        if i >= 15:  # Show top 15
            break
            
        fdr_result = fdr_dict[bc_combo]
        bonf_result = bonf_dict.get(bc_combo, fdr_result)  # Use FDR result if not in Bonf
        
        bc_display = bc_combo[:33] + ".." if len(bc_combo) > 35 else bc_combo
        
        fdr_sig_symbol = "✓" if fdr_result['significant_corrected'] else "✗"
        bonf_sig_symbol = "✓" if bonf_result['significant_corrected'] else "✗"
        
        print(f"{bc_display:35} | {fdr_result['correlation']:>7.3f} | "
              f"{fdr_result['p_value']:>10.4f} | "
              f"{fdr_result['p_adjusted']:>10.4f} | "
              f"{bonf_result['p_adjusted']:>10.4f} | "
              f"{fdr_sig_symbol:>5} | {bonf_sig_symbol:>5}")
    
    if len(results_fdr) > 15:
        print(f"\n... and {len(results_fdr) - 15} more BC combinations")
    
    # Show the most robust findings
    if robust_findings:
        print("\n" + "="*100)
        print("MOST ROBUST BC COMBINATIONS (survive both corrections)")
        print("="*100)
        
        # Sort robust findings by correlation strength
        robust_sorted = sorted(robust_findings, 
                             key=lambda x: abs(fdr_dict[x]['correlation']), 
                             reverse=True)
        
        for i, bc in enumerate(robust_sorted[:5], 1):  # Show top 5 most robust
            result = fdr_dict[bc]
            bonf_result = bonf_dict[bc]
            print(f"\n{i}. BC: '{bc}'")
            print(f"   Correlation: {result['correlation']:.3f}")
            print(f"   Effect size: {'Large' if abs(result['correlation']) >= 0.5 else 'Medium' if abs(result['correlation']) >= 0.3 else 'Small'}")
            print(f"   Raw p-value: {result['p_value']:.6f}")
            print(f"   FDR-adjusted p: {result['p_adjusted']:.6f}")
            print(f"   Bonferroni-adjusted p: {bonf_result['p_adjusted']:.6f}")
            
            # Count occurrences
            convs_with_bc = sum(1 for x in result['x_values'] if x > 0)
            total_occurrences = sum(result['x_values'])
            print(f"   Appears in: {convs_with_bc}/{len(result['x_values'])} conversations")
            print(f"   Total occurrences: {total_occurrences}")
    
    # Show findings that are significant with FDR but not Bonferroni
    if fdr_only:
        print("\n" + "="*100)
        print("BC COMBINATIONS SIGNIFICANT WITH FDR BUT NOT BONFERRONI")
        print("(These may still be meaningful but require more caution)")
        print("="*100)
        
        fdr_only_sorted = sorted(fdr_only, 
                                key=lambda x: abs(fdr_dict[x]['correlation']), 
                                reverse=True)
        
        for bc in fdr_only_sorted[:5]:  # Show top 5
            result = fdr_dict[bc]
            bonf_result = bonf_dict[bc]
            print(f"\nBC: '{bc}'")
            print(f"  Correlation: {result['correlation']:.3f}")
            print(f"  Raw p-value: {result['p_value']:.6f}")
            print(f"  FDR-adjusted p: {result['p_adjusted']:.6f} (< 0.05) ✓")
            print(f"  Bonferroni-adjusted p: {bonf_result['p_adjusted']:.6f} (> 0.05) ✗")
    
    # Statistical power analysis
    print("\n" + "="*100)
    print("STATISTICAL POWER CONSIDERATIONS")
    print("="*100)
    print(f"With {len(results_fdr)} tests:")
    print(f"  • Bonferroni requires p < {0.05/len(results_fdr):.6f} for significance")
    print(f"  • This is {len(results_fdr)}x more stringent than nominal α=0.05")
    print(f"  • FDR controls the expected proportion of false discoveries")
    print(f"  • FDR is more appropriate for exploratory analyses like this")
    
    if len(bonf_sig) == 0:
        print("\n⚠️  No findings survive Bonferroni correction!")
        print("   This is common with many comparisons and suggests:")
        print("   1. The study may be underpowered for such strict correction")
        print("   2. Effect sizes may be modest")
        print("   3. FDR is more appropriate for this exploratory analysis")
    
    # Save comparison results to CSV
    comparison_df = pd.DataFrame()
    for bc_combo in set([r['bc_combination'] for r in results_fdr]):
        fdr_result = next((r for r in results_fdr if r['bc_combination'] == bc_combo), None)
        bonf_result = next((r for r in results_bonf if r['bc_combination'] == bc_combo), None)
        
        if fdr_result:
            comparison_df = pd.concat([comparison_df, pd.DataFrame([{
                'bc_combination': bc_combo,
                'correlation': fdr_result['correlation'],
                'raw_p_value': fdr_result['p_value'],
                'fdr_adjusted_p': fdr_result['p_adjusted'],
                'bonf_adjusted_p': bonf_result['p_adjusted'] if bonf_result else None,
                'significant_fdr': fdr_result['significant_corrected'],
                'significant_bonferroni': bonf_result['significant_corrected'] if bonf_result else False,
                'robust_finding': bc_combo in robust_findings
            }])], ignore_index=True)
    
    comparison_df = comparison_df.sort_values('correlation', key=abs, ascending=False)
    comparison_df.to_csv('correction_methods_comparison.csv', index=False)
    print("\n💾 Comparison results saved to: correction_methods_comparison.csv")
    
    return results_fdr, results_bonf, robust_findings

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
    
    print("Statistical Analysis with Multiple Testing Correction")
    print("=" * 60)
    
    # Method 1: Use corrected analysis
    print("\nMethod 1: Full Dataset with Multiple Testing Correction")
    results_corrected = analyze_all_bc_combinations_corrected(
        df_last, df_metadata,
        pattern_col="BC Pattern T2",
        speaker="counsellor",
        min_occurrences=5,
        correction_method='fdr_bh'
    )
    
    # Use new summary function
    print_corrected_summary(results_corrected)
    effect_size_analysis(results_corrected)
    
    # ADD THIS NEW SECTION - BONFERRONI COMPARISON
    print("\n" + "=" * 100)
    print("COMPARING FDR VS BONFERRONI CORRECTIONS")
    print("=" * 100)
    results_fdr, results_bonf, robust_findings = compare_correction_methods(
        df_last, df_metadata,
        pattern_col="BC Pattern T2",
        speaker="counsellor",
        min_occurrences=5
    )
    
    # OPTIONAL: Add train/test validation
    print("\n" + "=" * 60)
    print("Method 2: Train/Test Split Validation")
    validation_results = train_test_validation(
        df_last, df_metadata,
        pattern_col="BC Pattern T2",
        speaker="counsellor",
        min_occurrences=5
    )
    
    if validation_results:
        train_results, test_results = validation_results
        
        print("\nValidation Results:")
        print("-" * 40)
        for test_result in test_results:
            bc = test_result['bc_combination']
            train_r = test_result.get('train_correlation', 'N/A')
            test_r = test_result['correlation']
            train_sig = test_result.get('train_significant', False)
            test_sig = test_result['p_value'] < 0.05
            
            print(f"BC: {bc[:40]}")
            print(f"  Train correlation: {train_r:.3f} (significant: {train_sig})")
            print(f"  Test correlation:  {test_r:.3f} (significant: {test_sig})")
            print(f"  Replicates: {'Yes' if abs(test_r) >= 0.1 and test_sig else 'No'}")
            print()
    
    # VOLLEY ANALYSIS - WITHOUT PRINTING TO TERMINAL
    print("\n" + "=" * 100)
    print("DETAILED VOLLEY ANALYSIS")
    print("=" * 100)
    
    # Set print_volleys=False to suppress terminal output (DEFAULT)
    # Set print_volleys=True if you want to see volleys in terminal
    volley_report = create_detailed_volley_report(
        df_full, df_last, df_metadata,
        pattern_col="BC Pattern T2",
        speaker="counsellor",
        output_file="significant_bc_volleys.csv",
        top_n=10,
        volleys_per_bc=None,  # Get all volleys
        print_volleys=False   # 🔹 This suppresses volley printing to terminal
    )
    
    # Specific BC analysis - also suppressed
    print("\n" + "=" * 100)
    print("DETAILED ANALYSIS OF SPECIFIC BC COMBINATIONS")
    print("=" * 100)
    
    if results_corrected:
        most_significant = results_corrected[0]
        bc_to_analyze = most_significant['bc_combination']
        
        print(f"\nMost significant BC combination: '{bc_to_analyze}'")
        volley_details = find_volleys_for_conversations_with_bc(
            df_full, df_last, most_significant, bc_to_analyze,
            pattern_col="BC Pattern T2", speaker="counsellor",
            print_volleys=False  # 🔹 This also suppresses volley printing
        )
        
        if volley_details is not None:
            print(f"Found {len(volley_details)} volleys with this BC combination")
            # Save this specific analysis too
            volley_details.to_csv("most_significant_bc_volleys.csv", index=False)
            print("💾 Specific BC volley data saved to: most_significant_bc_volleys.csv")
    
    # Optional: Plot top 3 most correlated BC combinations
    print("\nGenerating plots for top 3 most correlated BC combinations...")
    for i, result in enumerate(results_corrected[:3]):
        plot_bc_regression(result, save_path=f"bc_regression_{i+1}.png")
    
    # Optional: Analyze BC diversity
    print("\n" + "=" * 100)
    print("ANALYZING BC DIVERSITY")
    print("=" * 100)
    analyze_total_bc_diversity(df_last, df_metadata, pattern_col="BC Pattern T2", speaker="counsellor")
    
    print("\n🎉 Analysis complete! Check the CSV files for detailed volley data.")
    
    return volley_report


if __name__ == "__main__":
    volley_data = main()