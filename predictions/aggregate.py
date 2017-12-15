"""
This file contains functions necessary to aggregate the scores of 
the model trained with different randomg seed values.

"""

import pandas as pd
import os,sys


def read_fn(fn):
    print("Reading file: ", fn)
    return pd.read_csv(fn)

def main(fns):
    data = [read_fn(fn) for fn in fns]
    max_data = []
    
    for didx, d in enumerate(data):
        d.drop(d.columns[[-2, -1]], axis=1, inplace=True) # dropping the columns with the loss
        df_norm = (d - d.min()) / (d.max() - d.min()) # normalizing the DFrame
        df_norm['mean'] = df_norm.mean(axis=1) # getting weighted avg for each row 
        m_idx = df_norm['mean'].idxmax() # getting the idx of the row with max avg
        m_avg = df_norm['mean'][m_idx] # getting the max avg
        m_scores = d.iloc[m_idx] # retrieving the corresponding metric scores
        max_data.append(m_scores)
        
    md = pd.DataFrame(max_data)
    return md



if __name__ == '__main__':
    fns = sys.argv[1:]
    maxes_across_runs = main(fns)
    maxes_norm = (maxes_across_runs - maxes_across_runs.min()) / (maxes_across_runs.max() - maxes_across_runs.min())
    maxes_across_runs['mean'] = maxes_norm.mean(axis=1)
    
    print("\n--- Best scores for each run\n", maxes_across_runs)
    print("\n--- Scores statistics:")
    print(maxes_across_runs.describe())    
