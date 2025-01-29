import pandas as pd
import os

def convert_csv_to_tsv(input_dir, output_dir):
    """
    Convert all CSV files in input directory to TSV format.
    Uses only junction_aa as TCR sequence and clonal_frequency as Abundance.
    
    Args:
        input_dir (str): Path to directory containing CSV files
        output_dir (str): Path to directory where TSV files will be saved
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    for csv_file in csv_files:
        try:
            input_path = os.path.join(input_dir, csv_file)
            output_file = csv_file.replace('.csv', '.tsv')
            output_path = os.path.join(output_dir, output_file)
            
            print(f"\nProcessing: {csv_file}")
            
            # Read the CSV file
            df = pd.read_csv(input_path)
            
            # Create new dataframe with only required columns
            df_final = pd.DataFrame({
                'TCR': df['junction_aa'],
                'Abundance': df['clonal_frequency']
            })
            
            # Save to TSV file
            df_final.to_csv(output_path, sep='\t', index=False)
            print(f"Successfully converted to: {output_file}")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

# Specify your input and output directories here
input_directory = "/scratch/project/tcr_ml/BertTCR/ZERO_raw_csv"
output_directory = "ZERO_raw_tsv"

# Run the conversion
convert_csv_to_tsv(input_directory, output_directory)