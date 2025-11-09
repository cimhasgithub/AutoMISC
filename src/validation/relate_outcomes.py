from scipy.stats import linregress, pearsonr, spearmanr
from omegaconf import DictConfig
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import os
from pathlib import Path
from statsmodels.api import OLS, add_constant
import seaborn as sns
from hydra.utils import log
import hydra
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import wilcoxon


def get_slope(session: pd.DataFrame, normalize: bool=True):
    client_utt = session[session["speaker"] == "client"]
    y = client_utt['score'].to_numpy()
    if len(y) < 2:
        return None
    x = 100 * session.index.get_indexer(client_utt.index) / (len(session) - 1) if normalize else np.arange(len(y))
    result = linregress(x, y)
    return result

def get_ruler_scores(input_dataset: str, prolific_id: str, delta_with: str, target_variable: str) -> tuple:
    fn = ""
    if input_dataset == "MIV6.3A":
        fn = 'data/demographics/6.3A_demographics.csv'
    elif input_dataset == "MIV6.3B":
        fn = 'data/demographics/6.3B_demographics.csv'

    ruler_df = pd.read_csv(fn)
    match = ruler_df[ruler_df["Participant id"] == prolific_id]
    if match.empty:
        return None, None, None  # No match found
    pre = match[f"pre_{target_variable}"].values[0]
    post = match[f"{delta_with}_{target_variable}"].values[0]
    delta = post - pre
    return pre, post, delta

def compute_row(input_dataset, group, delta_with="week_later", target_variable="confidence"):
    pid = group['conv_id'].iloc[0]
    result = get_slope(group, normalize=True)
    pre, post, delta = get_ruler_scores(input_dataset, pid, delta_with, target_variable)

    c = (group["t1_label_auto"] == "C").sum()
    s = (group["t1_label_auto"] == "S").sum()
    r = group["t2_label_auto"].isin(["SR", "CR"]).sum()
    q = (group["t1_label_auto"] == "Q").sum()

    mico_codes = ['AF', 'ADP', 'EC', 'RCP', 'SU', 'OQ', 'CR', 'SR']
    miin_codes = ['ADW', 'CO', 'DI', 'RCW', 'WA']
    mico = group["t2_label_auto"].isin(mico_codes).sum()
    miin = group["t2_label_auto"].isin(miin_codes).sum()

    percent_mico = mico / (mico + miin) if (mico + miin) > 0 else 0
    rq = r / q if q > 0 else 0
    percent_ct = c / (c + s) if (c + s) > 0 else 0

    return pd.Series({
        "conv_id": pid,
        f'pre_{target_variable}': pre,
        f'{delta_with}_{target_variable}': post,
        f'delta_{target_variable}': delta,
        "slope": result.slope,
        "intercept": result.intercept,
        "r_value": result.rvalue,
        "%MIC": percent_mico,
        "R:Q": rq,
        "%CT": percent_ct,
    })

def compute_row_noconf(group):
    pid = group['conv_id'].iloc[0]
    c = (group["t1_label_auto"] == "C").sum()
    s = (group["t1_label_auto"] == "S").sum()
    r = group["t2_label_auto"].isin(["SR", "CR"]).sum()
    q = (group["t1_label_auto"] == "Q").sum()

    mico_codes = ['AF', 'ADP', 'EC', 'RCP', 'SU', 'OQ', 'CR', 'SR']
    miin_codes = ['ADW', 'CO', 'DI', 'RCW', 'WA']
    mico = group["t2_label_auto"].isin(mico_codes).sum()
    miin = group["t2_label_auto"].isin(miin_codes).sum()

    percent_mico = mico / (mico + miin) if (mico + miin) > 0 else np.nan
    rq = r / q if q > 0 else 0
    percent_ct = c / (c + s) if (c + s) > 0 else 0

    return pd.Series({
        "conv_id": pid,
        "%MIC": percent_mico,
        "R:Q": rq,
        "%CT": percent_ct,
    })

def linear_regression(df, input_dataset, delta_with="week_later", target_variable="delta_confidence"):
    # Derive the raw feature column (e.g., 'week_later_confidence')
    target_baseline_col = f"{delta_with}_{target_variable.split('_')[1]}"

    results = (
        df.groupby("conv_id", sort=False, group_keys=False)
        .apply(lambda group: compute_row(input_dataset, group, delta_with=delta_with, target_variable=target_variable.split('_')[1]))
        .dropna(subset=[target_variable])
    )

    exclude = {target_baseline_col, "conv_id", "r_value", "intercept"}

    corr_results = []
    for col in results.columns:
        if col in exclude or col == target_variable:
            continue
        if pd.api.types.is_numeric_dtype(results[col]):
            x = results[target_variable]
            y = results[col]
            if x.isna().any() or y.isna().any():
                continue
            r, p = spearmanr(x, y)
            corr_results.append((col, r, p))

    corr_df = pd.DataFrame(corr_results, columns=["variable", "spearman_r", "p_value"])

    # Save outputs
    exp_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    os.makedirs(exp_output_dir / "results", exist_ok=True)

    corr_df.to_csv(exp_output_dir / f'results/correlation_results_{target_variable}.csv', index=False)
    results.to_csv(exp_output_dir / f'results/agg_{target_variable}.csv')
    log.info(f"Saved aggregated results to {exp_output_dir / f'results/agg_{target_variable}.csv'}")
    log.info(f"Saved correlation results to {exp_output_dir / f'results/correlation_results_{target_variable}.csv'}")

    # Linear model + plot
    r, p = spearmanr(results['slope'], results[target_variable])
    X = add_constant(results['slope'])
    y = results[target_variable]
    model = OLS(y, X).fit()
    print(model.summary())

    plt.figure(figsize=(6, 4))
    plt.scatter(results['slope'], results[target_variable], alpha=0.8)
    pred_line = model.predict(X)
    plt.plot(results['slope'], pred_line, 'r--', label=f'Regression Line\nr={r:.2f}, p={p:.4f}')
    plt.xlabel('Motivation Slope')
    plt.ylabel(f'Change in {target_variable.split("_")[1].capitalize()} ({delta_with})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(exp_output_dir / f'results/motivation_vs_{target_variable}.pdf', format="pdf", bbox_inches='tight')
    plt.show()
    log.info(f"Saved plot to {exp_output_dir / f'results/motivation_vs_{target_variable}.pdf'}")

    return results, corr_df

def loocv(df, input_dataset, delta_with="week_later"):
    results = (
        df.groupby("conv_id", sort=False, group_keys=False)
        .apply(lambda group: compute_row(input_dataset, group, delta_with=delta_with))
        .dropna(subset=["delta_confidence"])
    )
    # features = ['slope', 'pre_confidence', '%MIC', 'R:Q', '%CT']
    features = ['slope']
    X = results[features].values
    y = results['delta_confidence'].values
    conv_ids = results['conv_id'].values

    loo = LeaveOneOut()
    y_pred, y_true, test_ids = [], [], []

    print("\n=== LOOCV PREDICTION SUMMARY ===")
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)[0]
        y_pred.append(y_hat)
        y_true.append(y_test[0])
        test_ids.append(conv_ids[test_idx][0])
        # print(f"Test PID: {conv_ids[test_idx][0]:<10} | slope: {X[test_idx][0][0]:>6.3f} | "
        #       f"true Δconf: {y_test[0]:>+5.2f} | predicted: {y_hat:>+5.2f} | error: {y_hat - y_test[0]:>+5.2f}")

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"\nFinal R²: {r2:.3f}, MSE: {mse:.3f}\n")

    # Plot predictions vs true values
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Fit")
    plt.xlabel("True Δ Confidence")
    plt.ylabel("Predicted Δ Confidence")
    plt.title(f"LOOCV: slope → Δ Confidence\nn={len(y)}, R²={r2:.2f}, MSE={mse:.2f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    err = np.array(y_pred) - np.array(y_true)
    with np.errstate(divide='ignore', invalid='ignore'):
        percent_err = np.where(np.abs(y_true) > 1e-5, (err / np.array(y_true)) * 100, np.nan)


    res = pd.DataFrame({
        "conv_id": test_ids,
        **dict(zip(features, X.T)),
        "true_delta": y_true,
        "predicted_delta": y_pred,
        "error": err,
        "percent_error": percent_err,
    })
    exp_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    res.to_csv(exp_output_dir / 'results/loocv_results.csv', index=False)
    log.info(f"Saved LOOCV results to {exp_output_dir / 'results/loocv_results.csv'}")
    return res

def plot_trajectory(df: pd.DataFrame, results: pd.DataFrame, pids: list[str], normalize: bool=True, delta_with: str="week_later"):

    plt.figure(figsize=(8, 4))
    for i, pid in enumerate(pids):
        session = df[df['conv_id'] == pid]
        client_utt = session[session["speaker"] == "client"]
        y = client_utt['score'].to_numpy()
        x = 100 * session.index.get_indexer(client_utt.index) / (len(session) - 1) if normalize else np.arange(len(y))

        session_result = results[results['conv_id'] == pid]
        if session_result.empty:
            print(f"No results found for conv_id: {pid}")
            continue
        # print(session_result)
        slope = session_result['slope'].iloc[0]
        intercept = session_result['intercept'].iloc[0]
        # r_value = session_result['r_value'].iloc[0]
        # pre = session_result['pre_confidence'].iloc[0]
        # post = session_result[f'{delta_with}_confidence'].iloc[0]
        delta = session_result['delta_confidence'].iloc[0]
        # plt.plot(x, y, 'o-', label = f"ID{i+1}: Δconf={int(delta):+} ({int(pre)}→{int(post)}), slope={slope:+.2f}")
        # plt.plot(x, y, '-', label = f"Δconf={int(delta):+}, slope={slope:+.2f}", color='lightgray', linewidth=1, zorder=1)
        plt.plot(x, y, 'o-', label = f"Δconf={int(delta):+}", linewidth=1, zorder=1)
        plt.xlim(-1, 101)

        line = intercept + slope * np.array(x)
        color = 'green' if slope > 0 else 'red'
        plt.plot(x, line, '--', color=color, linewidth=2, zorder=2)
        # print('plotted line')

    plt.xlabel('% Session Progress' if normalize else 'Utterance Index', fontsize=14)
    plt.ylabel('Motivation Score', fontsize=14)
    plt.ylim(-2.2,2.2)
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.grid(True)
    plt.legend(fontsize=16, loc='best')
    plt.tight_layout()
    exp_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    plt.savefig(exp_output_dir / 'results/sample_traj.pdf', format="pdf", bbox_inches='tight')
    plt.show()
    log.info(f"Saved sample trajectories plot to {exp_output_dir / 'results/sample_traj.pdf'}")

def confidence_histogram():
    version_A = pd.read_csv('outputs/2025-08-11/15-06-36/results/agg.csv')
    version_B = pd.read_csv('outputs/2025-08-11/15-08-19/results/agg.csv')

    bins = list(range(-5, 11))  # Only show -5 to 10
    width = 0.4

    # Histogram counts
    counts_A, _ = np.histogram(version_A['delta_confidence'], bins=bins)
    counts_B, _ = np.histogram(version_B['delta_confidence'], bins=bins)

    total_A = len(version_A)
    total_B = len(version_B)
    percent_A = counts_A / total_A * 100
    percent_B = counts_B / total_B * 100

    bin_centers = np.array(bins[:-1]) + 0.5
    positions_A = bin_centers - width / 2
    positions_B = bin_centers + width / 2

    # Colors based on sign
    colors_A = ['#0044cc' if x < 0 else '#99bbff' for x in bin_centers]
    colors_B = ['#cc7700' if x < 0 else '#ffcc99' for x in bin_centers]

    plt.figure(figsize=(10, 5))
    bars_A = plt.bar(positions_A, percent_A, width=width, label='MIBot v6.3A', color='blue', alpha=0.7)
    bars_B = plt.bar(positions_B, percent_B, width=width, label='MIBot v6.3B', color='orange', alpha=0.7)
    for bar, x in zip(bars_A, bin_centers):
        if x < 0:
            bar.set_hatch('//')
    
    for bar, x in zip(bars_B, bin_centers):
        if x < 0:
            bar.set_hatch('//')

    for bar, pct in zip(bars_A, percent_A):
        if pct > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{pct:.0f}',
                     ha='center', va='bottom', fontsize=10)
    for bar, pct in zip(bars_B, percent_B):
        if pct > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{pct:.0f}',
                     ha='center', va='bottom', fontsize=10)

    plt.axvline(x=0, color='grey', linestyle='dotted', linewidth=1)

    plt.xlabel('Change in Confidence (1-Week Later--Before)')
    plt.ylabel('Percentage of Participants')
    plt.ylim(0, 30)
    # plt.title('Distribution of Change in Confidence')
    plt.xticks(bin_centers, [str(int(x-0.5)) for x in bin_centers])
    legend_elements = [
        Patch(facecolor='blue', label='MIBot v6.3A', alpha=0.7),
        Patch(facecolor='orange', label='MIBot v6.3B', alpha=0.7)
    ]
    plt.legend(handles=legend_elements)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xlim(-5.5, 10.5)
    plt.tight_layout()

    exp_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    os.makedirs(exp_output_dir / "results", exist_ok=True)
    out_path = exp_output_dir / 'results/confidence_histogram.pdf'
    plt.savefig(out_path, format="pdf", bbox_inches='tight')
    plt.show()
    log.info(f"Saved confidence histogram plot to {out_path}")

def violin_plots():
    miv63a = pd.read_csv('data/annotated/MIV6.3A_lowconf_tiered_gpt-4.1-2025-04-14_interval_10_annotated.csv')
    miv63b = pd.read_csv('data/annotated/MIV6.3B_lowconf_tiered_gpt-4.1-2025-04-14_interval_3_annotated.csv')
    hlqc = pd.read_csv('data/annotated/datasets/HLQC_annotated.csv')
    hlqc_hi = hlqc[hlqc['conv_id'].str.startswith('high')].copy()
    hlqc_lo = hlqc[hlqc['conv_id'].str.startswith('low')].copy()

    # Compute metrics for each group
    df_hi = hlqc_hi.groupby('conv_id', sort=False, group_keys=False).apply(compute_row_noconf)
    df_lo = hlqc_lo.groupby('conv_id', sort=False, group_keys=False).apply(compute_row_noconf)
    df_a = miv63a.groupby('conv_id', sort=False, group_keys=False).apply(compute_row_noconf)
    df_b = miv63b.groupby('conv_id', sort=False, group_keys=False).apply(compute_row_noconf)

    # Add group labels
    df_hi["Group"] = "HLQC_HI"
    df_lo["Group"] = "HLQC_LO"
    df_a["Group"] = "MIv6.3A"
    df_b["Group"] = "MIv6.3B"

    df_all = pd.concat([df_lo, df_hi, df_a, df_b], ignore_index=True)
    exp_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    os.makedirs(exp_output_dir / "results", exist_ok=True)
    df_all.to_csv(exp_output_dir / "results/violin_plots_data.csv", index=False)
    log.info(f"Saved violin plots data to {exp_output_dir / 'results/violin_plots_data.csv'}")
    
    summary_stats = (
        df_all
        .groupby("Group")[["%MIC", "R:Q", "%CT"]]
        .agg(['mean', 'std'])
        .round(3)
    )

    # Flatten multi-index columns
    summary_stats.columns = [f"{metric}_{stat}" for metric, stat in summary_stats.columns]

    # Reset index and save
    summary_stats.reset_index(inplace=True)
    summary_stats_path = exp_output_dir / "results/violin_plot_summary_stats.csv"
    summary_stats.to_csv(summary_stats_path, index=False)
    log.info(f"Saved violin summary stats to {summary_stats_path}")

    # Group order and colors
    groups = ['HLQC_LO', 'HLQC_HI', 'MIv6.3A', 'MIv6.3B']
    colors = ['red', 'green', 'blue', 'orange']

    def get_grouped_data(metric):
        return [df_all[df_all["Group"] == g][metric].dropna().values for g in groups]

    fig, axes = plt.subplots(1, 3, figsize=(30, 5))

    data_mic = get_grouped_data('%MIC')
    vp1 = axes[0].violinplot(data_mic, showmeans=False, showmedians=False, showextrema=False, widths=0.8)
    for patch, color in zip(vp1['bodies'], colors): patch.set_facecolor(color); patch.set_alpha(0.4)

    data_rq = get_grouped_data('R:Q')
    vp2 = axes[1].violinplot(data_rq, showmeans=False, showmedians=False, showextrema=False, widths=0.8)
    for patch, color in zip(vp2['bodies'], colors): patch.set_facecolor(color); patch.set_alpha(0.4)
    axes[1].set_ylim(-0.5, 8.5)

    data_ct = get_grouped_data('%CT')
    vp3 = axes[2].violinplot(data_ct, showmeans=False, showmedians=False, showextrema=False, widths=0.8)
    for patch, color in zip(vp3['bodies'], colors): patch.set_facecolor(color); patch.set_alpha(0.4)

    # Common formatting
    for i, ax in enumerate(axes):
        # ax.set_xticks(range(1, len(groups)+1), labels=groups)
        ax.set_xticklabels(groups)
        ax.set_ylabel(["Percent MI-Consistent Responses (%MIC)", "Reflection to Question Ratio (R:Q)", "Percent Client Change Talk(%CT)"][i])
        metric = ["%MIC", "R:Q", "%CT"][i]
        # ax.set_ylabel(metric, fontsize=20)
        current_data = get_grouped_data(metric)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        bp = ax.boxplot(current_data, positions=range(1, len(groups) + 1), widths=0.2, patch_artist=True, 
                     showmeans=True, meanline=False, flierprops={'marker': '.', 'color': 'black'}, 
                     meanprops={'marker': '^', 'markerfacecolor': 'black', 'markeredgecolor': 'black'})
        for box in bp['boxes']:
            box.set(color='black', linewidth=1)  # Box outline
            box.set(facecolor='white')             # Box fill
        for whisker in bp['whiskers']:
            whisker.set(color='black', linewidth=1)  # Whiskers
        for cap in bp['caps']:
            cap.set(color='black', linewidth=1)  # Caps
        for median in bp['medians']:
            median.set(color='red', linewidth=2)  # Median line
        for mean in bp['means']:
            mean.set(marker='^', color='black', markersize=8)  # Mean triangle marker

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='^', color='black', markersize=8, linestyle='None', label='Mean'),
        Line2D([0], [0], color='red', linewidth=2, label='Median')
    ]
    for i in [0, 2]:
        axes[i].legend(handles=legend_elements, loc='best')
        axes[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))

    plt.tight_layout()

    out_path = exp_output_dir / 'results/violin_plots_matplotlib.pdf'
    plt.savefig(out_path, format="pdf", bbox_inches='tight')
    # plt.show()
    log.info(f"Saved matplotlib violin plots to {out_path}")

    for i, ax in enumerate(axes):
    # Compute tight bounding box for just this axis
        renderer = fig.canvas.get_renderer()
        tight_bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        
        # Save the cropped figure
        fig.savefig(exp_output_dir / f"violin_panel_{i+1}.pdf", bbox_inches=tight_bbox, format='pdf')

def stacked_bar_plot():
    # Load datasets
    miv63a = pd.read_csv('data/annotated/MIV6.3A_lowconf_tiered_gpt-4.1-2025-04-14_interval_10_annotated.csv')
    miv63b = pd.read_csv('data/annotated/MIV6.3B_lowconf_tiered_gpt-4.1-2025-04-14_interval_3_annotated.csv')
    hlqc = pd.read_csv('data/annotated/datasets/HLQC_annotated.csv')

    # Split HLQC into LO and HI
    hlqc_hi = hlqc[hlqc['conv_id'].str.startswith('high')].copy()
    hlqc_lo = hlqc[hlqc['conv_id'].str.startswith('low')].copy()

    # Define datasets
    datasets = {
        'HLQC_LO': hlqc_lo,
        'HLQC_HI': hlqc_hi,
        'MIV6.3A': miv63a,
        'MIV6.3B': miv63b
    }

    # Label sets
    counsellor_labels = ['IMI', 'IMC', 'SRL', 'CRL', 'Q', 'O']
    client_labels = ['C', 'S', 'N']
    label_colors = {
        'IMI': '#1f77b4', 'IMC': '#ff7f0e', 'SRL': '#2ca02c',
        'CRL': '#d62728', 'Q': '#9467bd', 'O': '#8c564b',
        'C': '#e377c2', 'S': '#7f7f7f', 'N': '#bcbd22'
    }

    # Collect counts
    rows = []
    for dataset_name, df in datasets.items():
        for speaker, labels in [('counsellor', counsellor_labels), ('client', client_labels)]:
            counts = df[df['speaker'] == speaker]['t1_label_auto'].value_counts(normalize=True) * 100
            row = {'Dataset': dataset_name, 'Speaker': speaker}
            for label in labels:
                row[label] = counts.get(label, 0)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.set_index(['Dataset', 'Speaker'], inplace=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.35
    x = range(len(df.index.levels[0]))  # one per dataset

    for i, (dataset, group_df) in enumerate(df.groupby(level=0)):
        for j, speaker in enumerate(['counsellor', 'client']):
            xpos = i + (j - 0.5) * bar_width
            row = df.loc[(dataset, speaker)]
            label_set = counsellor_labels if speaker == 'counsellor' else client_labels
            bottom = 0
            for label in label_set:
                height = row.get(label, 0)
                if height > 0:
                    ax.bar(
                        xpos, height, width=bar_width, bottom=bottom,
                        color=label_colors[label],
                        edgecolor='white',
                        label=label if (i == 0 and j == 0 and label in counsellor_labels) or
                                       (i == 0 and j == 1 and label in client_labels) else None
                    )
                    if height > 3:
                        ax.text(xpos, bottom + height / 2, f'{height:.0f}%', ha='center', va='center', fontsize=10, color='white')
                    bottom += height

    # X-axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(df.index.levels[0])
    ax.set_ylabel('Percentage of Utterances (%)')
    # ax.set_title('Distribution of Tier 1 Labels by Dataset and Speaker')
    # ax.legend(title='T1 Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    handles, labels = ax.get_legend_handles_labels()

    # Manually insert "headers"
    handles = ([plt.Line2D([0], [0], color='none')] + handles[:6] +
            [plt.Line2D([0], [0], color='none')] + handles[6:])
    labels = (['Counsellor:'] + labels[:6] + ['Client:'] + labels[6:])

    # Legend with no indent on headers
    legend = ax.legend(
        handles, labels,
        title='T1 Label',
        loc='upper left', bbox_to_anchor=(1.02, 1),
        handlelength=1.5, handletextpad=0.5,
    )
    legend.get_title().set_fontweight('bold')

    # Make "Counsellor:" and "Client:" bold & remove bullet spacing
    for text in legend.get_texts():
        if text.get_text() in ['T1 Label', 'Counsellor:', 'Client:']:
            text.set_fontweight('bold')
            text.set_horizontalalignment('left')


    # ax.legend(handles, labels, title='T1 Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    exp_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    os.makedirs(exp_output_dir / "results", exist_ok=True)
    plt.savefig(exp_output_dir / 'results/t1_label_distribution.pdf', format="pdf", bbox_inches='tight')
    log.info(f"Saved T1 label distribution plot to {exp_output_dir / 'results/t1_label_distribution.pdf'}")
    plt.show()

def CARE_analysis() -> None:
    v52  = pd.read_csv('data/demographics/5.2_demographics.csv')
    v63a = pd.read_csv('data/demographics/6.3A_demographics.csv')
    v63b = pd.read_csv('data/demographics/6.3B_demographics.csv')

    v52 = pd.read_csv('data/demographics/5.2_demographics.csv')
    v63a = pd.read_csv('data/demographics/6.3A_demographics.csv')
    v63b = pd.read_csv('data/demographics/6.3B_demographics.csv')

    datasets = {
        "MIBot v5.2": (v52, 'purple'),
        "MIBot v6.3A": (v63a, 'blue'),
        "MIBot v6.3B": (v63b, 'orange')
    }

    # === 1. Histogram of total CARE scores (percentage) ===
    bin_edges = np.arange(0, 55, 5)  # bins: 0-5, 6-10, ..., 46-50
    bin_labels = [f"{i+1}–{i+5}" for i in bin_edges[:-1]]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = 1.5

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (label, (df, color)) in enumerate(datasets.items()):
        counts, _ = np.histogram(df["CARE"], bins=bin_edges)
        percentages = counts / counts.sum() * 100
        ax.bar(bin_centers + i * width - width, percentages, width=width, label=label, color=color, alpha=0.7)

    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels, rotation=0)
    # ax.set_xlabel("Total CARE Score")
    ax.set_ylabel("Percentage of Participants")
    # ax.set_title("Distribution of CARE Scores")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    exp_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    os.makedirs(exp_output_dir / "results", exist_ok=True)
    plt.savefig(exp_output_dir / 'results/care_score_histogram.pdf', format="pdf", bbox_inches='tight')
    log.info(f"Saved CARE score histogram to {exp_output_dir / 'results/care_score_histogram.pdf'}")
    plt.show()

    # === 2. Mean CARE score per question ===
    question_labels = [f"CAREQ{i}" for i in range(1, 11)]
    x = np.arange(len(question_labels))
    bar_width = 0.25

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for i, (label, (df, color)) in enumerate(datasets.items()):
        means = df[question_labels].mean()
        ax2.bar(x + i * bar_width, means, width=bar_width, label=label, color=color, alpha=0.7)

    ax2.set_xticks(x + bar_width)
    ax2.set_xticklabels([q[-2:] if q != "CAREQ10" else "Q10" for q in question_labels ])
    ax2.set_ylabel("Mean CARE Score")
    # ax2.set_title("Mean CARE Score per Question")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(exp_output_dir / 'results/mean_care_score_per_question.pdf', format="pdf", bbox_inches='tight')
    log.info(f"Saved mean CARE score per question to {exp_output_dir / 'results/mean_care_score_per_question.pdf'}")
    plt.show()

def demographic_compare() -> None:
    v52 = pd.read_csv('data/demographics/5.2_demographics.csv')
    v63a = pd.read_csv('data/demographics/6.3A_demographics.csv')
    v63b = pd.read_csv('data/demographics/6.3B_demographics.csv')
    v52['Version'] = '5.2'
    v63a['Version'] = '6.3A'
    v63b['Version'] = '6.3B'
    df = pd.concat([v52, v63a, v63b], ignore_index=True)

    # Define demographic fields to summarize
    demo_fields = {
        'Sex': ['Male', 'Female'],
        'Age': 'rass',  # adjust bin labels to match your data
        'Ethnicity simplified': ['White', 'Black', 'Asian', 'Mixed', 'Other'],
        'Student status': ['Yes', 'No', 'DATA_EXPIRED'],
        'Employment status': ['Full-Time', 'Part-Time', "Not in paid work (e.g. homemaker', 'retired or disabled)", 'Unemployed (and job seeking)', 'Other', 'DATA_EXPIRED'],
        'Country of residence': None,  # will dynamically select top 5
        'Country of birth': None,      # will dynamically select top 5
    }

    # Result accumulator
    results = []

    for field, categories in demo_fields.items():
        if categories is None:
            # Determine top 5 then label others as "Other"
            top5_res = df[field].value_counts().nlargest(7).index
            df[field + '_grouped'] = df[field].apply(lambda x: x if x in top5_res else 'Other')
            group_field = field + '_grouped'
            categories = list(top5_res) + ['Other']
        elif field == 'Age':
            # Define custom age bins and labels
            bins = [0, 19, 29, 39, 49, 59, 69, 79, float('inf')]
            labels = ['Below 20', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80 or above']
            df['Age_grouped'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True, include_lowest=True)
            group_field = 'Age_grouped'
            categories = labels
        else:
            group_field = field

        # Count and percentage for each version
        summary = (
            df.groupby(['Version', group_field])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=categories, fill_value=0)
        )

        # Add percentages
        percent = summary.div(summary.sum(axis=1), axis=0) * 100
        percent = percent.round(0).astype(int).astype(str)

        # Combine counts and percentages into one string
        combined = summary.astype(str) + ' (' + percent + ')'
        # combined['Attribute'] = field
        # results.append(combined)

        combined = combined.transpose()
        combined['Attribute'] = field
        combined = combined.set_index('Attribute', append=True).reorder_levels([1, 0])
        results.append(combined)

    # Concatenate all and reorder
    final_df = pd.concat(results)
    # final_df = final_df.set_index('Attribute', append=True).reorder_levels([1, 0])
    # final_df = final_df.sort_index()

    # Save to CSV
    final_df.to_csv('data/demographics/demographic_comparison.csv')

    print("Saved to data/demographics/demographic_comparison.csv")

def demographic_compare_confidence() -> None:
    # Load CSVs
    v52 = pd.read_csv('data/demographics/5.2_demographics.csv')
    v63a = pd.read_csv('data/demographics/6.3A_demographics.csv')
    v63b = pd.read_csv('data/demographics/6.3B_demographics.csv')
    v52['Version'] = '5.2'
    v63a['Version'] = '6.3A'
    v63b['Version'] = '6.3B'
    df = pd.concat([v52, v63a, v63b], ignore_index=True)

    # Filter to those with valid confidence data
    conf_cols = ['pre_confidence', 'post_confidence', 'week_later_confidence', 'delta_confidence']
    df = df.dropna(subset=conf_cols)

    # Create demographic groups
    df['Age'] = df['Age'].apply(lambda x: '<30' if x < 30 else '30+')
    df['Ethnicity'] = df['Ethnicity simplified'].apply(lambda x: 'White' if x == 'White' else 'Other')
    df['Employment status'] = df['Employment status'].apply(lambda x: 'Full-Time' if x == 'Full-Time' else 'Other')

    # Prepare comparison groups
    groups = {
        'Sex': ['Male', 'Female'],
        'Age': ['<30', '30+'],
        'Ethnicity': ['White', 'Other'],
        'Employment status': ['Full-Time', 'Other'],
    }

    records = []
    version_totals = df['Version'].value_counts().to_dict()

    for demo_field, values in groups.items():
        for value in values:
            for version in ['5.2', '6.3A', '6.3B']:
                subset = df[(df['Version'] == version) & (df[demo_field] == value)]
                n = len(subset)
                if n == 0:
                    continue  # Skip empty groups

                # Compute means and SDs
                pre_m, pre_sd = subset['pre_confidence'].mean(), subset['pre_confidence'].std()
                post_m, post_sd = subset['post_confidence'].mean(), subset['post_confidence'].std()
                week_m, week_sd = subset['week_later_confidence'].mean(), subset['week_later_confidence'].std()
                delta_m, delta_sd = subset['delta_confidence'].mean(), subset['delta_confidence'].std()

                # Format with 1 decimal place, mean (SD)
                def fmt(mean, sd):
                    return f"{round(mean, 1)} ({round(sd, 1)})"

                total_n = version_totals[version]
                percent = round(n / total_n * 100)
                count_str = f"{n} ({percent}%)"

                # Wilcoxon signed-rank test (one-sided: week > pre)
                try:
                    stat, p_value = wilcoxon(
                        subset['week_later_confidence'],
                        subset['pre_confidence'],
                        alternative='greater'
                    )
                    # p_value = round(p_value, 3)
                except ValueError:
                    p_value = 'NA'

                records.append({
                    'Demo. Factor': demo_field,
                    'Value': value,
                    'Version': version,
                    'Count (%)': count_str,
                    'Pre-confidence': fmt(pre_m, pre_sd),
                    'Post-confidence': fmt(post_m, post_sd),
                    'Week-later confidence': fmt(week_m, week_sd),
                    'Delta (Week - Pre)': fmt(delta_m, delta_sd),
                    'p (one-sided Wilcoxon)': p_value,
                })

    # Create and save summary table
    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv('data/demographics/confidence_by_group.csv', index=False)
    print("Saved to data/demographics/confidence_by_group.csv")

def relate_outcomes(cfg: DictConfig) -> None:
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 14

    auto_anno_path = Path('data/annotated') / (
        f"{cfg.input_dataset.name}_"
        f"{cfg.input_dataset.subset}_"
        f"{cfg.annotated.class_structure}_"
        f"{cfg.annotated.model.rsplit('/', 1)[-1]}_"
        f"{cfg.annotated.context_mode}_"
        f"{cfg.annotated.num_context_turns if cfg.annotated.context_mode == 'interval' else ''}" 
        f"_annotated.csv"
        # f"_annotated_full.csv"
    )
    log.info(f"AutoMISC annotations path: {auto_anno_path}")
    df = pd.read_csv(auto_anno_path)
    label_to_score = {
        "TS-": -2, "AC-": -2, "C-": -2, 
        "N-": -1, "R-": -1, "AB-": -1, "D-": -1, 
        "O-": 0,
        "N": 0, 
        "O+": 0,
        "D+": 1, "AB+": 1, "R+": 1, "N+": 1, 
        "C+": 2, "AC+": 2, "TS+": 2, 
    }
    df['score'] = df['t2_label_auto'].map(label_to_score)

    results, _ = linear_regression(df, cfg.input_dataset.name, delta_with="week_later", target_variable="delta_confidence")
    # linear_regression(df, cfg.input_dataset.name, delta_with="week_later", target_variable="delta_importance")
    # linear_regression(df, cfg.input_dataset.name, delta_with="week_later", target_variable="delta_readiness")

    # loocv(df, input_dataset=cfg.input_dataset.name, delta_with="week_later")

    # confidence_histogram()
    # violin_plots()
    # CARE_analysis()
    # demographic_compare()
    # demographic_compare_confidence()
    # stacked_bar_plot()

    # fn_A = 'outputs/2025-08-09/19-05-39/results/agg_2.csv'
    # fn_B = 'outputs/2025-08-09/18-30-09/results/agg.csv'
    # df_A = pd.read_csv(fn_A)
    # df_B = pd.read_csv(fn_B)
    # from scipy.stats import ttest_ind
    # for metric in ['delta_confidence', '%MIC', 'R:Q', '%CT', 'slope']:
    #     t_stat, p_value = ttest_ind(df_A[metric], df_B[metric], equal_var=False)
    #     log.info(f"Statistical test results between A and B on {metric}: t-statistic = {t_stat}, p-value = {p_value}")


    # PLOTTING TRAJECTORIES
    ids_to_plot = [
        '6615283f82799fd70e2c056b', 
                   '5e19b24b2bf9512392d6dd90',
                   ]
    # ids_to_plot = df['conv_id'].unique()
    print(ids_to_plot)
    plot_trajectory(df, results, ids_to_plot, normalize=True, delta_with="week_later")

