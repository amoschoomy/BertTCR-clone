import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('/scratch/project/tcr_ml/BertTCR/seekgene_results.csv')

# Create the boxplot
plt.figure(figsize=(14, 10))
plt.boxplot(df['Average_Probability'])

# Customize the plot
plt.title('Distribution of Average Probabilities')
plt.ylabel('Probability')
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot
plt.savefig('probability_boxplot.png', dpi=300, bbox_inches='tight')

# Close the figure to free memory
plt.close()

# Print summary statistics
print("\nSummary Statistics:")
print(df['Average_Probability'].describe())