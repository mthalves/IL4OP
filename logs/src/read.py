import numpy as np
import pandas as pd

def planning(nexp:int, method:str, path:str, env:str, 
    columns:list = ['time','reward','time2reason'],
    time_constrained:bool=False, max_time_minutes:float=10.):

    time_index = columns.index('time')
    reward_index = columns.index('reward')
    last_task_time = 0.0

    final_results = []
    for exp in range(nexp):
        # reading the data
        results = []
        with open(path+method+'_'+env+'_'+str(exp)+'.csv','r') as resultfile:
            line_count = 0
            for line in resultfile:
                if line_count > 0: # skipping header
                    # splitting results
                    results.append([])
                    fcolumns = line.strip().split(';')
                    fcolumns = [c for c in fcolumns if c.strip() != '']
                    
                    # checking time
                    if time_constrained:
                        if float(fcolumns[time_index]) > max_time_minutes*60:
                            break

                    # appending manipulated data if needed
                    if 'time_per_task' in columns:
                        if float(fcolumns[reward_index]) > 0:
                            fcolumns.append(0.0)
                            last_task_time = fcolumns[time_index]
                        else:
                            fcolumns.append(float(fcolumns[time_index])-float(last_task_time))

                    # appending result
                    for i in range(len(columns)):
                        results[-1].append(float(fcolumns[i]))

                line_count += 1
        
        # post-processing
        # - aggregating time per second
        df = pd.DataFrame(results, columns=columns)
        df["time"] = df["time"].astype(int)

        # aggregate per second
        agg = df.groupby("time").mean()

        # build full range of seconds from min to max
        full_index = pd.RangeIndex(0, df["time"].max() + 1)

        # reindex to include missing seconds (NaN where no data exists)
        agg = agg.reindex(full_index, fill_value=0).reset_index()
        agg = agg.rename(columns={"index": "time"})

        final_results.append(agg)
    return final_results