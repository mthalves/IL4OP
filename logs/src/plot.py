import logs.src.stats as sts
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

FIG_COUNTER = 0
FIGSIZE = (6.4,5.4)

FONTSIZE = 28
LEGEND_FONTSIZE = 12
FONT_DICT = {
        'weight': 'bold',
        'size': 26,
        }
TICK_FONTSIZE = 20

COLOR_VEC   = ['tab:blue','tab:green','tab:red','tab:orange','tab:purple','tab:brown','tab:pink','tab:olive']
            
MARKER_SIZE = 18
MARK_EVERY = 20
MARKER_VEC = ['o','^','p','s','X','o']

LINEWIDTH = 5
LINESTYLE_VEC = ['--','-',':','-.','-.']

COLOR_DICT = {
    'POMCP':'tab:blue',
    'IB-POMCP':'tab:orange',
    'TB ρ-POMCP':'#9467db', #tab:purple'
    'TB ρ-POMCP (2s)':'#9467db', #tab:purple'
    'TB ρ-POMCP (1s)':"#c96cd7",
    'ρ-POMCP':'tab:brown',
}
MARKER_DICT = {
    'POMCP':'o',
    'IB-POMCP':'^',
    'ρ-POMCP':'p',
    'TB ρ-POMCP':'s',
    'TB ρ-POMCP (2s)':'s',
    'TB ρ-POMCP (1s)':'s',
}
LINESTYLE_VEC_DICT = {
    'pomcp':'--',
    'POMCP':'--',
    'ibpomcp':'-',
    'IB-POMCP':'-',
    'rhopomcp':':',
    'ρ-POMCP':':',
    'tbrhopomcp':'-.',
    'TB ρ-POMCP':'-.',
    'TB ρ-POMCP (2s)':'-.',
    'TB ρ-POMCP (1s)':'-.',
}

def lines(
 results:dict,
 target_data:str,
 ylabel:str='y-axis',xlabel:str='x-axis',
 cum_sum:bool=True,
 save:bool=False,savepath:str='./plots/',
 env_name:str='',
 fixed_max_len: int | None = None,
 complete_with: str = 'zero'):
    global FIG_COUNTER, FIGSIZE
    plt.figure(num=FIG_COUNTER,figsize=FIGSIZE)

    y = {}
    y_lower = {}
    y_upper = {}
    counter = 0
    for method in results:
        # calculating the mean and confidence intervals
        y[method], y_lower[method], y_upper[method] =\
            sts.by_iteration(
                results[method],
                complete_with=complete_with,
                cumsum=cum_sum,
                fixed_max_len=fixed_max_len)
        print(len(y[method]))
        # plotting
        plt.fill_between(
            range(len(y[method])),
            y_lower[method][target_data],
            y_upper[method][target_data],
            color=COLOR_DICT[method],alpha=0.4)

        MARK_EVERY = int((len(y[method]))/15)
        plt.plot(
            range(len(y[method])),
            y[method][target_data],label=method,
            color=COLOR_DICT[method],
            marker=MARKER_DICT[method], markersize=MARKER_SIZE,markevery=MARK_EVERY,
            linewidth=LINEWIDTH,linestyle=LINESTYLE_VEC_DICT[method], markeredgecolor='black')
        counter += 1

    #plt.legend(loc='best',ncol=1,fontsize=18,edgecolor='black')
    plt.xlabel(xlabel,fontdict=FONT_DICT)
    plt.xticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.ylabel(ylabel,fontdict=FONT_DICT)
    plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)
    b, t = plt.ylim()
    plt.ylim(0,t)
    plt.tight_layout()

    if save:
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        plt.savefig(savepath+env_name+'_'+target_data+'_lines.pdf')
    else:
        plt.show()
    FIG_COUNTER += 1

def bars(
 results:dict,
 target_data:str,
 ylabel:str='y-axis',
 save:bool=False,savepath:str='./plots/',
 env_name:str='',
 fixed_max_len: int | None = None,
 complete_with: str = 'zero'):
    global FIG_COUNTER, FIGSIZE
    plt.figure(num=FIG_COUNTER,figsize=FIGSIZE)

    y = {}
    y_lower = {}
    y_upper = {}

    xticks = []
    heights = []
    errors = []
    colors = []

    counter = 0
    for method in results:
        # calculating the mean and confidence intervals
        y[method], y_lower[method], y_upper[method] =\
            sts.by_experiment(
                results[method],
                target_data=target_data,
                complete_with=complete_with,
                cumsum=True,
                fixed_max_len=fixed_max_len)
        
        xticks.append(counter)
        heights.append(y[method])
        errors.append((y_upper[method]-y_lower[method])/2)
        colors.append(COLOR_DICT[method])

        counter += 1
    
    plt.bar(xticks,heights,yerr=errors,
            width=0.8,align='center',
            color=colors,edgecolor='black',
            linewidth=1, tick_label=y.keys(),capsize=5)
    #plt.legend(loc='best',ncol=1,fontsize=18,edgecolor='black')
    plt.xticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.ylabel(ylabel,fontdict=FONT_DICT)
    plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)
    b, t = plt.ylim()
    plt.ylim(0,t)
    plt.tight_layout()

    
    if save:
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        plt.savefig(savepath+env_name+'_'+target_data+'_bars.pdf')
    else:
        plt.show()
    FIG_COUNTER += 1

def boxes(
 results: dict,
 target_data: str,
 ylabel: str = 'y-axis',
 save: bool = False,
 savepath: str = './plots/',
 env_name: str = '',
 fixed_max_len: int | None = None,
 complete_with: str = 'zero'):
    global FIG_COUNTER, FIGSIZE
    plt.figure(num=FIG_COUNTER, figsize=FIGSIZE)

    data = []
    labels = []
    colors = []

    for method in results:
        # collect all the raw results for the boxplot
        vals, _, _ = sts.by_iteration(
            results[method],
            complete_with=complete_with,
            cumsum=False,
            fixed_max_len=fixed_max_len
        )

        data.append(vals[target_data])
        labels.append(method)
        colors.append(COLOR_DICT[method])

    # Create the boxplot
    print(data)
    print(labels)
    bp = plt.boxplot(
        data,
        patch_artist=True,        # to allow box coloring
        notch=False,
        labels=labels,
        showmeans=True
    )

    # Color each box according to COLOR_DICT
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)

    # Style whiskers, caps, and medians
    for whisker in bp['whiskers']:
        whisker.set(color="black", linewidth=1)
    for cap in bp['caps']:
        cap.set(color="black", linewidth=1)
    for median in bp['medians']:
        median.set(color="black", linewidth=1.5)
    for mean in bp['means']:
        mean.set(marker="o", markerfacecolor="black", markeredgecolor="black")

    plt.xticks(fontsize=TICK_FONTSIZE, rotation=20)
    plt.ylabel(ylabel, fontdict=FONT_DICT)
    plt.yticks(fontsize=TICK_FONTSIZE, rotation=45)

    plt.tight_layout()

    if save:
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        plt.savefig(savepath + env_name + '_' + target_data + '_boxplot.pdf')
    else:
        plt.show()
    FIG_COUNTER += 1
