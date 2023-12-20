
# Import packages
import pandas as pd

# Read csv files
df = pd.read_csv('../winequality-red.csv')
qual= df['quality'].value_counts()
qual.to_csv('Quality_Count.csv')
