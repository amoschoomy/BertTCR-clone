#!/usr/bin/env python3
"""
Convert CSV file with AA_seq,CloneFreq columns to TSV file with TCR,Abundance columns.
Usage: python3 convert_to_tsv.py
"""

import pandas as pd
import os

def convert_csv_to_tsv():
    input_file = '/scratch/project/tcr_ml/iCanTCR/seekgene/S3.csv'
    output_file = '/scratch/project/tcr_ml/BertTCR/seekgene_raw_tsv/S3.tsv'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
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
        
        print(f"Successfully converted '{input_file}' to '{output_file}'")
        print(f"Processed {len(df)} records")
        return True
        
    except Exception as e:
        print(f"Error converting file: {str(e)}")
        return False

if __name__ == "__main__":
    convert_csv_to_tsv()