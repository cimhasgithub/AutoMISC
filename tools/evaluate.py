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


def majority_csn_from_pattern(pattern):
    """Determines the majority label from a comma-separated pattern of C/S/N."""
    if pd.isna(pattern):
        return np.nan

    labels = pattern.split(", ")
    c, s, n = labels.count("C"), labels.count("S"), labels.count("N")

    if c > s:
        return "C"
    elif s > c:
        return "S"
    elif c == s and c > 0:
        for label in reversed(labels):
            if label == "C":
                return "C"
            elif label == "S":
                return "S"
    elif c == 0 and s == 0 and n > 0:
        return "N"

    return np.nan


def plot_client_majority_distribution(client_df):
    """Plots the distribution of majority CSN categories for client volleys."""
    counts = client_df["Client Majority CS (T1)"].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=140)
    plt.title("Distribution of Client Majority CS (T1)")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_client_c_ratio_distribution(client_df):
    """Plots histogram of C / (C + S) ratios per session."""
    cs_only = client_df[client_df["Client Majority CS (T1)"].isin(["C", "S"])]
    session_c_rate = cs_only.groupby("conv_id")["Client Majority CS (T1)"].apply(
        lambda labels: (labels == "C").sum() / len(labels)
    )

    mean_c = session_c_rate.mean()
    median_c = session_c_rate.median()

    plt.figure(figsize=(10, 6))
    sns.histplot(session_c_rate, bins=20, kde=True, color="skyblue")
    plt.axvline(mean_c, color="red", linestyle="--", label=f"Mean: {mean_c:.2f}")
    plt.axvline(median_c, color="green", linestyle="-", label=f"Median: {median_c:.2f}")
    plt.xlabel("C / (C + S) per Session")
    plt.ylabel("Number of Sessions")
    plt.title("Distribution of Client C Ratio by Session")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_counsellor_word_count_distribution(df):
    """Plots the distribution of counsellor volley word counts."""
    lengths = df["vol_text"].dropna().apply(lambda text: len(str(text).split()))
    mean_len = lengths.mean()
    median_len = lengths.median()

    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, kde=True, bins=30, color="skyblue")
    plt.axvline(mean_len, color='red', linestyle='--', label=f'Mean: {mean_len:.2f}')
    plt.axvline(median_len, color='green', linestyle='--', label=f'Median: {median_len:.2f}')
    plt.xlabel("Word Count per Counsellor Volley")
    plt.ylabel("Number of Volleys")
    plt.title("Counsellor Volley Length Distribution (Word Count)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def display_pie_chart_examples(df, pattern_col, title, threshold, original_df):
    """Displays examples of each pie chart category in the terminal with BC labels."""
    print(f"\n{title} - Examples of Each Category:")
    print("=" * 80)
    
    counts = df[pattern_col].value_counts(normalize=True)
    main = counts[counts >= threshold]
    other = counts[counts < threshold]
    
    # Show examples for main categories
    for pattern in main.index:
        examples = df[df[pattern_col] == pattern][["corp_vol_idx", "vol_text"]].dropna().head(2)
        print(f"\nCategory: {pattern} ({main[pattern]:.1%})")
        print("-" * 50)
        for i, (_, row) in enumerate(examples.iterrows(), 1):
            corp_vol_idx = row["corp_vol_idx"]
            vol_text = row["vol_text"]
            
            # Get utterances for this volley from original data
            volley_utterances = original_df[original_df["corp_vol_idx"] == corp_vol_idx]
            
            print(f"Example {i}:")
            for _, utt_row in volley_utterances.iterrows():
                utt_text = utt_row["utt_text"]
                t1_label = utt_row["t1_label_auto"]
                t2_label = utt_row["t2_label_auto"]
                print(f"  • {utt_text} [{t1_label}] [{t2_label}]")
            print()
    
    # Show examples for "Other" category
    if other.sum() > 0:
        print(f"\nCategory: Other ({other.sum():.1%})")
        print("-" * 50)
        other_examples = df[df[pattern_col].isin(other.index)][["corp_vol_idx", "vol_text"]].dropna().head(2)
        for i, (_, row) in enumerate(other_examples.iterrows(), 1):
            corp_vol_idx = row["corp_vol_idx"]
            vol_text = row["vol_text"]
            
            # Get utterances for this volley from original data
            volley_utterances = original_df[original_df["corp_vol_idx"] == corp_vol_idx]
            
            print(f"Example {i}:")
            for _, utt_row in volley_utterances.iterrows():
                utt_text = utt_row["utt_text"]
                t1_label = utt_row["t1_label_auto"]
                t2_label = utt_row["t2_label_auto"]
                print(f"  • {utt_text} [{t1_label}] [{t2_label}]")
            print()
    
    print("=" * 80)


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


def perform_linear_regression_analysis(df):
    """Perform linear regression of average BCs per counsellor turn vs client change fraction."""
    conv_ids = sorted(df['conv_id'].unique())
    
    change_fractions = []
    avg_bcs_per_turn = []
    valid_conv_ids = []
    
    # Calculate metrics for each conversation
    for conv_id in conv_ids:
        change_frac = calculate_client_change_fraction(conv_id)
        avg_bcs = calculate_average_bcs_per_turn(df, conv_id)
        
        if change_frac is not None and avg_bcs is not None:
            change_fractions.append(change_frac)
            avg_bcs_per_turn.append(avg_bcs)
            valid_conv_ids.append(conv_id)
    
    if len(change_fractions) < 2:
        print("Not enough valid data points for regression analysis")
        return
    
    # Convert to numpy arrays for regression
    # Change fraction is independent variable (X), avg BCs is dependent variable (y)
    X = np.array(change_fractions).reshape(-1, 1)
    y = np.array(avg_bcs_per_turn)
    
    # Perform linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate statistics
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, y_pred)
    
    # Calculate correlation and p-value
    correlation, p_value = stats.pearsonr(change_fractions, avg_bcs_per_turn)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(change_fractions, avg_bcs_per_turn, alpha=0.6, s=100, label='Conversations')
    
    # Regression line
    plt.plot(change_fractions, y_pred, color='red', linewidth=2, label=f'Regression line')
    
    # Add conversation IDs as labels
    for i, conv_id in enumerate(valid_conv_ids):
        plt.annotate(f'{conv_id}', (change_fractions[i], avg_bcs_per_turn[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Client Change Fraction (C/(C+S))', fontsize=12)
    plt.ylabel('Average BCs per Counsellor Turn', fontsize=12)
    plt.title('Linear Regression: Average BCs per Turn vs Client Change Fraction', fontsize=14)
    
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
    print("\nLinear Regression Analysis Results:")
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
    print(f"{'Conv ID':>8} | {'Change Fraction':>15} | {'Avg BCs/Turn':>12}")
    print("-" * 50)
    for i, conv_id in enumerate(valid_conv_ids):
        print(f"{conv_id:>8} | {change_fractions[i]:>15.3f} | {avg_bcs_per_turn[i]:>12.2f}")


def main():
    df = read_and_preprocess_csv("/Users/joeberson/Developer/AutoMISC/data/annotated/random_install_multiBC_20convos_all_tiered_gpt-4o_interval_5_annotated.csv")
    df_last = filter_last_volley_rows(df)

    # Counsellor analysis
    counsellor_df = df_last[df_last["speaker"].str.lower() == "counsellor"]
    display_pie_chart_examples(counsellor_df, "BC Pattern T1", "Counsellor BC Pattern T1 Combinations", threshold=0.03, original_df=df)
    display_pie_chart_examples(counsellor_df, "BC Pattern T2", "Counsellor BC Pattern T2 Combinations", threshold=0.02, original_df=df)
    plot_bc_pattern_distribution(counsellor_df, "BC Pattern T1", "Counsellor BC Pattern T1 Combinations", threshold=0.03)
    plot_bc_pattern_distribution(counsellor_df, "BC Pattern T2", "Counsellor BC Pattern T2 Combinations", threshold=0.02)
    plot_counsellor_word_count_distribution(counsellor_df)

    # Client analysis
    df_last["Client Majority CS (T1)"] = df_last.apply(
        lambda row: majority_csn_from_pattern(row["BC Pattern T1"]) if row["speaker"].lower() == "client" else np.nan,
        axis=1
    )

    client_df = df_last[df_last["speaker"].str.lower() == "client"].copy()
    plot_client_majority_distribution(client_df)
    plot_client_c_ratio_distribution(client_df)
    
    # NEW: Linear regression analysis
    print("\nPerforming linear regression analysis...")
    perform_linear_regression_analysis(df)


if __name__ == "__main__":
    main()