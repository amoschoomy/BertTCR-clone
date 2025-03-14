import pandas as pd
import os

# Define input and output directories
INPUT_DIR = "/scratch/project/tcr_ml/iCanTCR/sarcoma_zero_input"
OUTPUT_DIR = "/scratch/project/tcr_ml/BertTCR/sarcoma_zero_input"

def convert_csv_to_tsv(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all CSV files in input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in '{input_dir}'.")
        return

    for csv_file in csv_files:
        input_file = os.path.join(input_dir, csv_file)
        output_file = os.path.join(output_dir, csv_file.replace('.csv', '.tsv'))

        try:
            # Read the CSV file
            df = pd.read_csv(input_file)

            # Rename the columns
            df = df.rename(columns={
                'AA_seq': 'TCR',
                'CloneFreq': 'Abundance'
            })

            # Save as TSV file
            df.to_csv(output_file, sep='\t', index=False)

            print(f"Converted '{input_file}' → '{output_file}' ({len(df)} records)")

        except Exception as e:
            print(f"Error processing '{input_file}': {str(e)}")

if __name__ == "__main__":
    convert_csv_to_tsv(INPUT_DIR, OUTPUT_DIR)
