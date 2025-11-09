import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import DictConfig
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, ConfusionMatrixDisplay
from collections import Counter
import seaborn as sns

def annoMI_row_to_codes(row):
    code_map = {
        'therapist_input': {
            'information': ['GI'],
            'advice': ['ADVI'],
            'options': ['OPTI'],
            'negotiation': ['NEGO'],
        },
        'reflection': {
            'simple': ['SR'],
            'complex': ['CR'],
        },
        'question': {
            'open': ['OQ'],
            'closed': ['CQ'],
        },
        'client_talk_type': {
            'change': ['C'],
            'sustain': ['S'],
            'neutral': ['N'],
        }
    }

    codes = []
    for cat, mapping in [('therapist_input', 'therapist_input_subtype'),
                         ('reflection', 'reflection_subtype'),
                         ('question', 'question_subtype')]:
        if row.get(f'{cat}_exists') is True:
            subtype = str(row.get(mapping, '')).lower()
            codes.extend(code_map[cat][subtype])

    if not any(row.get(f'{cat}_exists') is True for cat in ['therapist_input', 'reflection', 'question']) \
       and str(row.get('main_therapist_behaviour', '')).lower() == 'other':
        codes.extend(['OTHE'])
    codes.extend(code_map['client_talk_type'].get(str(row.get('client_talk_type', '')).lower(), []))
    return list(set(codes))

def majority_vote(code_lists):
    # Return all codes tied for the highest count
    all_codes = [code for sublist in code_lists for code in sublist if pd.notna(code)]
    if not all_codes:
        return []
    code_counts = Counter(all_codes)
    max_count = max(code_counts.values())
    majority = [code for code, count in code_counts.items() if count == max_count]
    return sorted(majority)

def break_tie(label_list):
    for pref in ['C', 'S', 'N']:
        if pref in label_list:
            return pref
    return sorted(label_list)[0] if label_list else None

def compare_annomi():
    # fn_auto = 'AnnoMI_annotated_tiered_gpt-4o-interval-10_annotated_old.csv'
    fn_auto = 'AnnoMI_lowconf_tiered_gpt-4.1-2025-04-14_interval_10_annotated.csv'
    utt_codes = pd.read_csv(Path('data/annotated') / fn_auto)
    grouped = utt_codes.groupby(['conv_id', 'corp_vol_idx'], sort=False)
    vol_codes_auto = grouped.agg({
        't2_label_auto': lambda x: list(x),
        'speaker': 'first',
        'vol_text': 'first'
    }).reset_index(drop=True)
    print(vol_codes_auto)

    fn_og = 'AnnoMI-full.csv'
    vol_codes_og = pd.read_csv(Path('data') / fn_og)
    vol_codes_og.rename(columns={"interlocutor": "speaker",
                                 'utterance_text': 'vol_text',
                                 'utterance_id': 'conv_vol_idx',
                                 'transcript_id': 'conv_id'}, inplace=True)
    vol_codes_og["speaker"] = vol_codes_og["speaker"].str.replace("therapist", "counsellor", case=False, regex=False)
    vol_codes_og['annomi_label'] = vol_codes_og.apply(annoMI_row_to_codes, axis=1)
    vol_codes_og = vol_codes_og.groupby(['conv_id', 'conv_vol_idx'], sort=False).agg({
        'annomi_label': lambda code_lists: majority_vote(code_lists),
        'speaker': 'first',
        'vol_text': 'first',
        'conv_vol_idx': 'first'
    }).reset_index(drop=True)
    print(vol_codes_og)
    
    assert len(vol_codes_auto) == len(vol_codes_og), "Mismatch in number of volleys between AutoMISC and original AnnoMI data."
    assert vol_codes_auto['speaker'].equals(vol_codes_og['speaker']), "Speakers do not match between AutoMISC and original AnnoMI data."

    merged = pd.concat([
        vol_codes_auto.drop(columns=['vol_text', 'speaker']),
        vol_codes_og
    ], axis=1)
    print(merged)

    annomi_to_auto = {
        'OQ': {'OQ'},
        'CQ': {'CQ'},
        'SR': {'SR'},
        'CR': {'CR', 'RF', 'AF', 'SU'},
        'GI': {'GI'},
        'ADVI': {'ADP', 'ADW'},
        'OPTI': {'ADP', 'ADW', 'EC', 'ST'},
        'NEGO': {'ADP', 'ADW', 'EC', 'RCP', 'RCW', 'WA', 'CO', 'DI', 'ST'},
        'OTHE': {'FA', 'FI', },
        'C': {'D+', 'AB+', 'R+', 'N+', 'C+', 'AC+', 'TS+', 'O+'},
        'S': {'D-', 'AB-', 'R-', 'N-', 'C-', 'AC-', 'TS-', 'O-'},
        'N': {'N'}
    }

    def compare_labels(row):
        predicted = set(row['t2_label_auto'])
        annomi_labels = row['annomi_label']
        required_sets = [annomi_to_auto.get(label, set()) for label in annomi_labels]
        # Require at least one predicted label in each required set
        match = all(len(predicted & req_set) > 0 for req_set in required_sets)
        matched = [label for label, req_set in zip(annomi_labels, required_sets) if predicted & req_set]
        missing = [label for label in annomi_labels if label not in matched]
        return {
            't2_label_auto_set': sorted(predicted),
            'match': match,
            'matched_labels': matched,
            'missing_labels': missing
        }

    comparison_results = merged.apply(compare_labels, axis=1, result_type='expand')
    final = pd.concat([merged, comparison_results], axis=1)
    print(final)
    final.to_csv('data/final.csv')

    # Summary
    match_rate = final['match'].mean()
    print(f"Match rate (all labels): {match_rate:.2%}")
    counsellor_df = final[final['speaker'] == 'counsellor']
    client_df = final[final['speaker'] == 'client']
    counsellor_acc = counsellor_df['match'].mean()
    print(f"Counsellor Match Rate: {counsellor_acc:.2%} (n={len(counsellor_df)})")
    client_acc = client_df['match'].mean()
    print(f"Client Match Rate: {client_acc:.2%} (n={len(client_df)})")

    def reduce_auto_label(auto_codes):
        label_mapping = {
            'D+': 'C', 'AB+': 'C', 'R+': 'C', 'N+': 'C', 'C+': 'C', 'O+': 'C', 'TS+': 'C', 'AC+': 'C',
            'D-': 'S', 'AB-': 'S', 'R-': 'S', 'N-': 'S', 'C-': 'S', 'O-': 'S', 'TS-': 'S', 'AC-': 'S',
            'N': 'N',
        }
        mapped = [label_mapping[c] for c in auto_codes if c in label_mapping]
        if not mapped:
            return None
        counts = Counter(mapped)
        max_count = max(counts.values())
        tied = [c for c, v in counts.items() if v == max_count]
        return break_tie(tied)

    client_df.loc[:, 'annomi_reduced'] = client_df['annomi_label'].apply(break_tie)
    client_df.loc[:, 'auto_reduced'] = client_df['t2_label_auto'].apply(reduce_auto_label)
    client_df.to_csv('data/client_reduced.csv', index=False)

    y_pred = client_df['annomi_reduced']
    y_true = client_df['auto_reduced'] 

    # Compute confusion matrix
    labels = ["C", "S", "N"]  # Ensure consistent label ordering
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
    # plt.xlabel('CodeCC')
    plt.xlabel('AutoMISC')
    plt.ylabel('AnnoMI')
    # plt.title('Client Label Confusion Matrix')
    plt.tight_layout()
    plt.savefig('data/test/annomi_automisc_client.pdf', bbox_inches='tight', format='pdf')
    plt.show()
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    print(f"Cohen's Kappa: {kappa:.3f}")

    accuracy = (y_true == y_pred).mean()
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return final


def compare(cfg: DictConfig) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20

    compare_annomi()

    return