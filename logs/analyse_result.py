import logs.src.plot as plt
import logs.src.read as readm
# if warnings are disturbing the presentation, uncomment the lines bellow
import warnings
warnings.filterwarnings("ignore")

# 1. Defining analysis  settings
NEXP = 50
ENV  = 'inspection'
SCENARIO = ['ushaped']
PATH = './logs/'+ENV+'/'

METHODS_DICT = {
    'pomcp':'POMCP',
    'ibpomcp':'IB-POMCP',
    'tbrhopomcp':'TB ρ-POMCP',
    'tbrhopomcp':'TB ρ-POMCP (2s)',
    'tbrhopomcp_1s':'TB ρ-POMCP (1s)',
}

SAVE         = True
SHOW_SUMMARY = True
SHOW_PVALUE  = True
PLOT         = True
PLOT_TYPE    = 'boxes' # 'lines', 'bars', 'boxes'

TARGET_DATA = 'time_per_task' # 'reward', 'time2reason', 'time_per_task

YLABEL_DICT = {
    'reward':'Cumulative Reward',
    'time2reason':'Time to Reason (s)',
    'time_per_task':'Spent Time (s)',
}
XLABEL = {
    'reward':'Execution Time (s)',
    'time2reason':'Execution Time (s)',
    'time_per_task':'Number of Tasks',
}
XLABEL, YLABEL = 'Execution Time (s)', YLABEL_DICT[TARGET_DATA]

# select the target methods
envs = [ENV+'_'+scenario for scenario in SCENARIO] if len(SCENARIO)>0 else [ENV]
methods = [name for name in METHODS_DICT]

all_results = {}
for env in envs:
    print('>',env)

    # 1. Reading the result files
    results = {}
    for method in methods:
        print(method,end=' ')
        results[METHODS_DICT[method]] = \
            readm.planning(nexp=NEXP,method=method,path=PATH,env=env,
                           columns=[
                               'time',
                               'reward',
                               'time2reason',
                               'time_per_task'])
        print(len(results[METHODS_DICT[method]]))
    all_results[env] = results
        
    # 2. Analysing via plot and pvalues
    if PLOT:
        if PLOT_TYPE == 'lines':
            plt.lines(results=all_results[env],fixed_max_len=900,
                    target_data=TARGET_DATA,ylabel=YLABEL,xlabel=XLABEL,
                    cum_sum=True,save=SAVE,savepath=PATH+'plots/',env_name=env)
        elif PLOT_TYPE == 'bars':
            plt.bars(results=all_results[env],
                    target_data=TARGET_DATA,ylabel=YLABEL,fixed_max_len=900,
                    save=SAVE,savepath=PATH+'plots/',env_name=env)
        elif PLOT_TYPE == 'boxes':
            plt.boxes(results=all_results[env],
                    target_data=TARGET_DATA,ylabel=YLABEL,
                    complete_with='last',
                    fixed_max_len=900,
                    save=SAVE,savepath=PATH+'plots/',env_name=env)
        else:
            raise NotImplementedError('Plot type not implemented.')
            
    if SHOW_SUMMARY:
        plt.sts.summary(results=all_results[env],
                complete_with='last',cum_sum=False,
                fixed_max_len=900,
                target_data=TARGET_DATA,LaTeX=True)

    if SHOW_PVALUE:
        plt.sts.pvalues(results=all_results[env],
                target_data=TARGET_DATA,
                complete_with='last',cumsum=True)