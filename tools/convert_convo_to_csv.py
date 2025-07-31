import os
import json
import re
import csv

def split_utterances(text):
    """Split text into utterances based on punctuation followed by a space."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def convertConversationToProlificCsv(path, prolific_id, output_dir, output_name, write_header=False):
    """Convert a conversation JSON to one row per volley (no utterance splitting)."""
    with open(path, 'r') as file:
        data = json.load(file)

    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, output_name)

    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["Prolific ID", "Speaker", "Volley #", "Volley"])

        for volley_counter, entry in enumerate(data):
            speaker = entry["name"]
            content = entry["content"].strip()
            writer.writerow([
                prolific_id,
                speaker,
                volley_counter,
                content
            ])

    return os.path.abspath(output_csv)

def convertConversationToProlificCsvUtterance(path, prolific_id,output_dir,output_name,write_header=False):
    """Convert a conversation JSON to rows in a CSV suitable for Prolific import. with utterance splitting"""
    with open(path, 'r') as file:
        data = json.load(file)
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, output_name)

    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["Prolific ID", "Speaker", "Volley #", "Volley", "Utterance #", "Utterance"])

        utterance_counter = 0
        volley_counter = 0

        for entry in data:
            speaker = entry["name"]
            utterances = split_utterances(entry["content"])
            cumulative = ""

            for utt in utterances:
                cumulative = (cumulative + " " + utt).strip()
                writer.writerow([
                    prolific_id,
                    speaker,
                    volley_counter,
                    cumulative,
                    utterance_counter,
                    utt
                ])
                utterance_counter += 1
            volley_counter += 1

    return os.path.abspath(output_csv)

# Main execution
if __name__ == "__main__":
    output_dir = os.path.join(os.getcwd(), "data")
    output_name = "random_install_multiBC_20convos.csv"
    for i in range(1, 20):
        conversation_json = f"/Users/joeberson/Developer/MIBot-v6/data/random_install_multiBC/real_sampled_client_{i}/conversation.json"
        convertConversationToProlificCsv(conversation_json, prolific_id=str(i), output_dir=output_dir,output_name=output_name, write_header=(i == 1))