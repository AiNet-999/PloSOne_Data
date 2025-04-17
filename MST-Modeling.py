import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

df = pd.read_csv("/content/SP500_Closing_Prices.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.fillna(method='ffill').fillna(method='bfill')
df=df.dropna()
total_records = len(df)
# remianing 30% data is not used , that is test set
short_cutoff = int(total_records * 0.2)
medium_cutoff = int(total_records * 0.4)
long_cutoff = int(total_records * 0.7)

short_term_df = df.iloc[:short_cutoff]
medium_term_df = df.iloc[:medium_cutoff]
long_term_df = df.iloc[:long_cutoff]

short_corr = short_term_df.corr()
medium_corr = medium_term_df.corr()
long_corr = long_term_df.corr()

short_adj = (short_corr.abs() >= 0.5).astype(int)
medium_adj = (medium_corr.abs() >= 0.5).astype(int)
long_adj = (long_corr.abs() >= 0.5).astype(int)

short_nodes = short_adj.shape[0]
medium_nodes = medium_adj.shape[0]
long_nodes = long_adj.shape[0]
#here we are calculting total edges without counting self-loop on node
short_edges = int((short_adj.values.sum() - short_nodes) / 2)
medium_edges = int((medium_adj.values.sum() - medium_nodes) / 2)
long_edges = int((long_adj.values.sum() - long_nodes) / 2)

print(f"Short-term (20%): {short_nodes} nodes, {short_edges} edges")
print(f"Medium-term (40%): {medium_nodes} nodes, {medium_edges} edges")
print(f"Long-term (70%): {long_nodes} nodes, {long_edges} edges")

total_ones_short = short_adj.values.sum()
total_ones_medium = medium_adj.values.sum()
total_ones_long = long_adj.values.sum()
#here total number 1's means total edges
print(f"Total 1s in short-term adjacency matrix: {total_ones_short}")
print(f"Total 1s in medium-term adjacency matrix: {total_ones_medium}")
print(f"Total 1s in long-term adjacency matrix: {total_ones_long}")


short_adj.to_csv("/content/short_term_adjacency.csv")
medium_adj.to_csv("/content/medium_term_adjacency.csv")
long_adj.to_csv("/content/long_term_adjacency.csv")