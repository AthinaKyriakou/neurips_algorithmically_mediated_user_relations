# Check the independence assumption
# To execute /opt/miniconda3/envs/user_intercon_env/bin/python /Users/athina/Desktop/Research/Experiments/User_Interconnectedness/20_check_independence_assumption.py

import pandas as pd
import numpy as np
from recommenders.datasets import movielens
import matplotlib.pyplot as plt
from scipy.stats import shapiro, wilcoxon, spearmanr, kendalltau
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from absl import flags
FLAGS = flags.FLAGS

def check_independence(type, algo_list, dataset, group_sizes, colors, markers, abs_influence=True):

    for algo in algo_list:
        if algo != 'MatrixFactorization_FunkSVD_Cython_unbiased' and algo != 'UserKNNCFRecommender':
            print("Algorithm %s not implemented" % algo)
            sys.exit()

    if type == 'scatter_plot':
        check_independence_scatter(algo_list, dataset, group_sizes, colors, markers, abs_influence=True)
    elif type == 'hist':
        check_independence_significance_hist(algo_list, dataset, group_sizes, colors)
    elif type == 'significance_testing':
        check_independence_significance_testing(algo_list, dataset, group_sizes)
    elif type == 'correlations':
        check_independence_correlations(algo_list, dataset, group_sizes)
    else:
        print("Type not considered")


def check_independence_significance_testing(algo_list, dataset, group_sizes):
    
    # path setup
    if dataset == 'MovieLens_100k':
        PRED_PATH = './result_experiments/MovieLens_100k/trainsets_' + str(FLAGS.ratio_split_train) + '/set_' + str(FLAGS.random_seed) + '/EvaluatorNegativeItemSample/'
    else:
        print("dataset not implemented")
        sys.exit()
    
    for algo in algo_list:
        pred_path_algo = PRED_PATH + algo + '/experiment_influence/'

        # 1. Read the actual group influences array
        fp_actual = pred_path_algo + 'group_influences_abs.csv'
        actual_influences_df = pd.read_csv(fp_actual, header=None)

        # 2. Read the additive group influences array
        fp_additive = pred_path_algo + 'group_influences_additive.csv'
        additive_influences_df = pd.read_csv(fp_additive, header=None)
    
        # 3. Significance testing: paired t-test between actual and additive group influences
        # # Check assumption: the sampling distribution of the differences is normal
        # # Null hypothesis of Sapiro-Wilk test: the data was drawn from a normal distribution
        # # --> reject if p < 0.05
        influences_diff_df = pd.DataFrame()
        print("\n\nChecking normality of differences")
        for i in range(0, len(group_sizes)):
            print('Group size:', group_sizes[i])
            influences_diff_df[i] = additive_influences_df[i] - actual_influences_df[i]
            stat1, p1 = shapiro(influences_diff_df[i]) # shapiro-wilk test
            print('Sapiro-Wilk test statistics=%.3f, p=%.3f' % (stat1, p1))
            #plt.hist(influences_diff_df[i])

        # Non-parametric Wilcoxon test to check significance of differences
        # https://people.umass.edu/bwdillon/files/linguist-609-2020/Notes/PairedSamples_Nonparametric.html#:~:text=The%20non%2Dparametric%20analog%20of,test%20would%20still%20be%20appropriate.
        # Null hypothesis of Wilcoxon test: the two related paired samples come from the same distribution
        # --> reject if p < 0.05 (yes!)
        influences_diff_df = pd.DataFrame()
        print("\n\nChecking significance of differences")
        for i in range(0, len(group_sizes)):
            print('Group size:', group_sizes[i])
            influences_diff_df[i] = additive_influences_df[i] - actual_influences_df[i]
            res = wilcoxon(influences_diff_df[i])
            print('Wilcoxon test statistics=%f, p=%f' % (res.statistic, res.pvalue))


def check_independence_scatter(algo_list, dataset, group_sizes, colors, markers, abs_influence=True):
    # path setup
    if dataset == 'MovieLens_100k':
        PRED_PATH = './result_experiments/MovieLens_100k/trainsets_' + str(FLAGS.ratio_split_train) + '/set_' + str(FLAGS.random_seed) + '/EvaluatorNegativeItemSample/'
    else:
        print("dataset not implemented")
        sys.exit()
    
    # Plotting
    fig, axes = plt.subplots(nrows=1, ncols=len(algo_list))
    for j in range (0, len(algo_list)):
        algo = algo_list[j]

        lims = []
        if algo == 'MatrixFactorization_FunkSVD_Cython_unbiased':
            lims = [0, 1000]
        elif algo == 'UserKNNCFRecommender':
            lims = [0, 18000]
        else:
            print("Algorithm %s not implemented" % algo)
            sys.exit()

        pred_path_algo = PRED_PATH + algo + '/experiment_influence/'

        # 1. Read the actual group influences array
        fp_actual = pred_path_algo + 'group_influences_abs.csv'
        actual_influences_df = pd.read_csv(fp_actual, header=None)

        # 2. Read the additive group influences array
        fp_additive = pred_path_algo + 'group_influences_additive.csv'
        additive_influences_df = pd.read_csv(fp_additive, header=None)

        # 3. Scatter plot
        for i in range(0, len(group_sizes)):
            if group_sizes[i] == '5':
                axes[j].scatter(actual_influences_df[i], additive_influences_df[i], label=group_sizes[i], c=colors[i], s=15, marker=markers[i], alpha=0.8)
            elif group_sizes[i] == '50':
                axes[j].scatter(actual_influences_df[i], additive_influences_df[i], label=group_sizes[i], c=colors[i], s=2, marker=markers[i])
            elif group_sizes[i] == '75':
                axes[j].scatter(actual_influences_df[i], additive_influences_df[i], label=group_sizes[i], c=colors[i], s=5, marker=markers[i])
            elif group_sizes[i] == '100':
                axes[j].scatter(actual_influences_df[i], additive_influences_df[i], label=group_sizes[i], c=colors[i], s=5, marker=markers[i], alpha=0.8)
            else: 
                axes[j].scatter(actual_influences_df[i], additive_influences_df[i], label=group_sizes[i], c=colors[i], s=20, marker=markers[i])
        
        axes[j].set_xlim(xmin=lims[0])
        axes[j].set_ylim(ymin=lims[0])
        axes[j].set_xlim(xmax=lims[1])
        axes[j].legend(loc='upper left', fancybox=True, ncol=2, fontsize=12)
        axes[j].tick_params(axis="x", labelsize=13, rotation=30)
        axes[j].tick_params(axis="y", labelsize=13)
                
    fig.tight_layout()
    #fig.suptitle('MovieLens 100k', fontsize=13)
    fig.supxlabel('$I_{relations}$', x=0.5, fontsize=18)
    fig.supylabel('$I_{independence}$', fontsize=18)    
    p = PRED_PATH + "check_independence_scatter.png"
    #plt.savefig(p, bbox_inches='tight', format="png", dpi=1200)
    plt.show()


def check_independence_significance_hist(algo_list, dataset, group_sizes, colors):
    
    # path setup
    if dataset == 'MovieLens_100k':
        PRED_PATH = './result_experiments/MovieLens_100k/trainsets_' + str(FLAGS.ratio_split_train) + '/set_' + str(FLAGS.random_seed) + '/EvaluatorNegativeItemSample/'
    else:
        print("dataset not implemented")
        sys.exit()
    
    fig, axes = plt.subplots(nrows=1, ncols=len(algo_list))
    for j in range (0, len(algo_list)):
        algo = algo_list[j]

        pred_path_algo = PRED_PATH + algo + '/experiment_influence/'

        # 1. Read the actual group influences array
        fp_actual = pred_path_algo + 'group_influences_abs.csv'
        actual_influences_df = pd.read_csv(fp_actual, header=None)

        # 2. Read the additive group influences array
        fp_additive = pred_path_algo + 'group_influences_additive.csv'
        additive_influences_df = pd.read_csv(fp_additive, header=None)

        for i in range(0, len(group_sizes)):
            axes[j].hist(additive_influences_df[i]-actual_influences_df[i], label=group_sizes[i], color=colors[i])
            axes[j].legend(loc='upper left', fancybox=True, ncol=2, fontsize=12)

    
    fig.tight_layout()
    #fig.suptitle('MovieLens 100k', fontsize=13)
    fig.supxlabel('$I_{independence}-I_{relations}$', x=0.5, fontsize=18)
    fig.supylabel('frequency', fontsize=18)    
    p = PRED_PATH + "check_independence_scatter.png"
    #plt.savefig(p, bbox_inches='tight', format="png", dpi=1200)
    plt.show()


def check_independence_correlations(algo_list, dataset, group_sizes):
    # path setup
    if dataset == 'MovieLens_100k':
        PRED_PATH = './result_experiments/MovieLens_100k/trainsets_' + str(FLAGS.ratio_split_train) + '/set_' + str(FLAGS.random_seed) + '/EvaluatorNegativeItemSample/'
    else:
        print("dataset not implemented")
        sys.exit()
    
    for algo in algo_list:
        pred_path_algo = PRED_PATH + algo + '/experiment_influence/'

        # 1. Read the actual group influences array
        fp_actual = pred_path_algo + 'group_influences_abs.csv'
        actual_influences_df = pd.read_csv(fp_actual, header=None)

        # 2. Read the additive group influences array
        fp_additive = pred_path_algo + 'group_influences_additive.csv'
        additive_influences_df = pd.read_csv(fp_additive, header=None)

        # 3. Check correlations per group size
        # # Check normality of influences: ok! not normal
        print('\n\nCheck normality of actual influences')
        for i in range(0, len(group_sizes)):
            print('Group size: ', group_sizes[i])
            stat1, p1 = shapiro(actual_influences_df[i])
            print('Sapiro-Wilk test statistics=%.3f, p=%.3f' % (stat1, p1))
            #plt.hist(actual_influences_unbiased_df[i])

        print('\n\nCheck normality of additive influences')
        for i in range(0, len(group_sizes)):
            print('Group size: ', group_sizes[i])
            stat1, p1 = shapiro(additive_influences_df[i])
            print('Sapiro-Wilk test statistics=%.3f, p=%.3f' % (stat1, p1))
            #plt.hist(additive_influences_unbiased_df[i])
       
        print("\n\nPer group size Spearman correlation")
        for i in range(0, len(group_sizes)):
            print('Group size: ', group_sizes[i])
            print('Spearman corr=%.3f' % (actual_influences_df[i].corr(additive_influences_df[i], method='spearman')))

        print("\n\nPer group size Kendall correlation")
        for i in range(0, len(group_sizes)):
            print('Group size: ', group_sizes[i])
            t = kendalltau(actual_influences_df[i], additive_influences_df[i]).correlation
            print('Kendall tau-b corr=%.3f' % t)