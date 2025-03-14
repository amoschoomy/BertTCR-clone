import pandas as pd
from collections import defaultdict

def process_tsv_batches(file_path):
    """
    Read and process a TSV file to merge batches and calculate average probabilities.
    
    Args:
        file_path (str): Path to the TSV file
    
    Returns:
        pandas.DataFrame: Processed data with merged batches and average probabilities
    """
    # Read the TSV file
    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # Create a dictionary to store probabilities for each base sample
    sample_data = defaultdict(list)
    
    # Process each row
    for _, row in df.iterrows():
        # Extract the base sample name (remove batch information)
        base_sample = row['Sample'].split('_batch_')[0]
        sample_data[base_sample].append(row['Probability'])
    
    # Calculate averages and create results
    results = []
    for sample, probabilities in sample_data.items():
        results.append({
            'Sample': sample,
            'Average_Probability': sum(probabilities) / len(probabilities),
            'Number_of_Batches': len(probabilities)
        })
    
    # Convert to DataFrame, sort by sample name, and round probabilities
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('Sample')
    result_df['Average_Probability'] = result_df['Average_Probability'].round(4)
    
    return result_df

def main():
    # Specify your TSV file path
    file_path = '/scratch/project/tcr_ml/BertTCR/sarcoma_ZERO_prediction.tsv'
    
    # Process the file
    results = process_tsv_batches(file_path)
    
    if results is not None:
        print("\nProcessed Results:")
        print(results.to_string(index=False))
        
        # Save the results to a CSV file
        output_file = 'sarcoma_ZERO_merged_batch_results.csv'
        results.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()