import numpy as np
import pandas as pd
import scipy.stats as stats
from itertools import combinations


def mean_confidence_interval(dfs, confidence=0.95, by_='iteration'):
    if by_ == 'iteration':
        # Stack all experiments into a 3D array: (n_experiments, n_rows, n_cols)
        arr = np.array([df.values for df in dfs])  
        
        # Compute mean across experiments
        mean = arr.mean(axis=0)
        
        # Standard error of the mean
        sem = stats.sem(arr, axis=0)
        
        # Degrees of freedom
        n = arr.shape[0]
        h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
        
        lower = mean - h
        upper = mean + h
        
        # Return as DataFrames with same columns/index as input
        mean_df = pd.DataFrame(mean, columns=dfs[0].columns, index=dfs[0].index)
        lower_df = pd.DataFrame(lower, columns=dfs[0].columns, index=dfs[0].index)
        upper_df = pd.DataFrame(upper, columns=dfs[0].columns, index=dfs[0].index)
    elif by_ == 'experiment':
        # Stack all experiments for this baseline
        arr = np.array([df.values for df in dfs])

        # Mean across experiments
        mean = arr.mean(axis=0)

        # Standard error of the mean
        sem = stats.sem(arr, axis=0)

        # Degrees of freedom
        n = arr.shape[0]
        h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)

        lower = mean - h
        upper = mean + h

        # Store results as DataFrames
        mean_df = pd.DataFrame(mean, columns=dfs[0].columns, index=dfs[0].index)
        lower_df = pd.DataFrame(lower, columns=dfs[0].columns, index=dfs[0].index)
        upper_df = pd.DataFrame(upper, columns=dfs[0].columns, index=dfs[0].index)
    else:
        raise NotImplementedError("by_ must be 'iteration' or 'experiment'")
    
    return mean_df, lower_df, upper_df

def by_iteration(
 data: list, 
 complete_with:str='zero',
 cumsum:bool=False, fixed_max_len:int|None=None):
    # 1. Formating data
    # finding the maximum length
    if not fixed_max_len:
        max_len = max([len(exp) for exp in data])
    else:
        max_len = fixed_max_len

    # building the y-axis data
    if complete_with=='zero':
        alligned =  [
            df.reindex(range(max_len), fill_value=0) 
            for df in data
        ]
    elif complete_with=='last':
        alligned =  [
            df.reindex(range(max_len), method='ffill').fillna(0) 
            for df in data
        ]
    else:
        alligned = data

    # retrieving the mean and the confidence interval
    if cumsum:
        alligned = [alligned[i].cumsum() for i in range(len(alligned))]
    m, l, u = mean_confidence_interval(alligned)
    return m, l, u

def by_experiment(
 data: list, 
 target_data:str='reward',
 complete_with:str='zero',
 cumsum:bool=False, fixed_max_len:int|None=None):
    # 1. Formating data
    # finding the maximum length
    if not fixed_max_len:
        max_len = max([len(exp) for exp in data])
    else:
        max_len = fixed_max_len

    # building the y-axis data
    if complete_with=='zero':
        alligned =  [
            df.reindex(range(max_len), fill_value=0) 
            for df in data
        ]
    elif complete_with=='last':
        alligned =  [
            df.reindex(range(max_len), fill_value=df[target_data].iloc[-1]) 
            for df in data
        ]
    else:
        alligned = data

    # retrieving the mean and the confidence interval
    if cumsum:
        alligned = [alligned[i].cumsum() for i in range(len(alligned))]
        
    m, l, u = mean_confidence_interval(alligned, by_='experiment')
    return m[target_data].mean(), l[target_data].mean(), u[target_data].mean()

def summary(
 results, 
 target_data, 
 complete_with:str='zero',
 cum_sum:bool=False,
 fixed_max_len:int|None=None,
 LaTeX=False):
    print('|||',target_data,'SUMMARY |||')
    for method in results:
        m, l, u = by_experiment(
            results[method],
            fixed_max_len=fixed_max_len,
            complete_with=complete_with,
            cumsum=cum_sum)
        ci = ((u-l)/2)
        
        if LaTeX:
            print(method,':\n$ %.3f \\pm %.3f $' % (m,ci))
        else:
            print(method,':',m,(u-l)/2)

def pvalues(results, target_data="reward", paired=True, 
    fixed_max_len=None, complete_with="zero", cumsum=False):
    """
    Compute pairwise p-values between baselines for a given metric, with alignment.

    Parameters
    ----------
    results : dict[str, list[pd.DataFrame]]
        Mapping from baseline name to list of experiment DataFrames.
    metric : str
        Column/metric to compare (e.g. "reward").
    paired : bool
        If True, run paired t-test (ttest_rel). Otherwise independent (ttest_ind).
    fixed_max_len : int or None
        Force all sequences to this length (if None, infer from max length).
    complete_with : {"zero", "nan"}
        How to pad shorter sequences ("zero" fills with 0, "nan" fills with NaN and ignores).
    cumsum : bool
        If True, take cumulative sum along time axis.

    Returns
    -------
    pd.DataFrame
        Square matrix of p-values between baselines.
    """
    baselines = list(results.keys())
    n = len(baselines)
    pval_matrix = pd.DataFrame(np.ones((n, n)), index=baselines, columns=baselines)

    # --- Step 1: Align all experiments ---
    all_aligned = {}
    for baseline, dfs in results.items():
        if not fixed_max_len:
            max_len = max(len(exp) for exp in dfs)
        else:
            max_len = fixed_max_len

        aligned = []
        for df in dfs:
            if complete_with == "zero":
                df_aligned = df.reindex(range(max_len), fill_value=0)
            elif complete_with == "last":
                df_aligned = df.reindex(range(max_len), fill_value=df[target_data].iloc[-1])
            elif complete_with == "nan":
                df_aligned = df.reindex(range(max_len))
            else:
                raise ValueError("complete_with must be 'zero' or 'nan'")

            if cumsum:
                df_aligned = df_aligned.cumsum()

            aligned.append(df_aligned)

        all_aligned[baseline] = aligned

    # --- Step 2: Pairwise statistical tests ---
    for a, b in combinations(baselines, 2):
        dfs_a = all_aligned[a]
        dfs_b = all_aligned[b]

        # take per-experiment mean of the chosen metric across time
        vals_a = np.array([df[target_data].mean(skipna=True) for df in dfs_a])
        vals_b = np.array([df[target_data].mean(skipna=True) for df in dfs_b])

        if paired:
            t_stat, p_val = stats.ttest_rel(vals_a, vals_b, nan_policy="omit")
        else:
            t_stat, p_val = stats.ttest_ind(vals_a, vals_b, nan_policy="omit")

        pval_matrix.loc[a, b] = p_val
        pval_matrix.loc[b, a] = p_val

    print(pval_matrix)