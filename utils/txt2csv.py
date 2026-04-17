import os
import re

import pandas as pd


def txt_to_csv_batch(input_folder: str, original_folder: str, output_folder: str):
    """Batch merge processed TXT files back into original CSV format."""
    os.makedirs(output_folder, exist_ok=True)

    txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    if not txt_files:
        print("No TXT files found.")
        return

    for txt_file in txt_files:
        txt_path = os.path.join(input_folder, txt_file)
        csv_name = os.path.splitext(txt_file)[0] + ".csv"
        csv_path = os.path.join(original_folder, csv_name)

        if not os.path.exists(csv_path):
            print(f"Original CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        speakers, texts = [], []
        for line in lines:
            match = re.match(r"^([^:]+)[:]\s*(.*)$", line)
            if match:
                speakers.append(match.group(1).strip())
                texts.append(match.group(2).strip())
            else:
                speakers.append("")
                texts.append(line.strip())

        if len(df) != len(speakers):
            print(f"Row count mismatch in {csv_name}: CSV={len(df)}, TXT={len(speakers)}. Truncating to shorter.")
            min_len = min(len(df), len(speakers))
            df = df.iloc[:min_len]
            speakers = speakers[:min_len]
            texts = texts[:min_len]

        df["speaker"] = speakers
        df["text"] = texts

        output_path = os.path.join(output_folder, csv_name)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Merged: {txt_file} -> {output_path}")

    print("Batch merge complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge processed TXT back into CSV")
    parser.add_argument("-i", "--input", required=True, help="Processed TXT folder")
    parser.add_argument("-r", "--reference", required=True, help="Original CSV folder")
    parser.add_argument("-o", "--output", required=True, help="Output CSV folder")
    args = parser.parse_args()
    txt_to_csv_batch(args.input, args.reference, args.output)
