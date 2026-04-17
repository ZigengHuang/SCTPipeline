import os
import random

import pandas as pd

random.seed(42)


def csv_to_txt_batch(input_folder: str, output_folder: str):
    """Batch convert CSV dialogue files to TXT format."""
    os.makedirs(output_folder, exist_ok=True)

    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
    if not csv_files:
        print("No CSV files found.")
        return

    for csv_file in csv_files:
        input_path = os.path.join(input_folder, csv_file)
        base_name = os.path.splitext(csv_file)[0]
        output_path = os.path.join(output_folder, base_name + ".txt")

        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            print(f"Failed to read {csv_file}: {e}")
            continue

        df["speaker"] = df["speaker"].astype(str)
        unique_speakers = df["speaker"].unique()
        roles = ["Patient", "Doctor"]
        speaker_map = {uid: random.choice(roles) for uid in unique_speakers}
        df["Role"] = df["speaker"].map(speaker_map)

        text_col = "transcription" if "transcription" in df.columns else "text"
        lines = []
        for _, row in df.iterrows():
            content = str(row.get(text_col, "")).strip()
            lines.append(f"{row['Role']}: {content}\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        print(f"Converted: {csv_file} -> {output_path} (mapping: {speaker_map})")

    print("Batch conversion complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert CSV dialogue files to TXT")
    parser.add_argument("-i", "--input", required=True, help="Input CSV folder")
    parser.add_argument("-o", "--output", required=True, help="Output TXT folder")
    args = parser.parse_args()
    csv_to_txt_batch(args.input, args.output)
