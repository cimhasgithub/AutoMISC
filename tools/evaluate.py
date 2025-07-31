import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


if __name__ == "__main__":
    main()
