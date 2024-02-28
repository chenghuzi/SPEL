import functools
import warnings
from collections import OrderedDict, defaultdict
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
from statannotations.Annotator import Annotator
import torch
from matplotlib_venn import venn3
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
from skimage.measure import block_reduce
from tqdm import tqdm

from analysis import (
    color_pal_trial_types,
    dt2phase_ids,
    extract_sim_history_effect,
    get_AB_decoding_accuracy,
    get_history_effect,
    get_neuron_ids4encoding,
    get_ws_target_ratio,
    plot_neuron_frs_across_conditions,
    preprocess_dataf,
    read_train_dynamics,
    get_collective_hs
)
from exp_utils import (
    filter_data,
)
from sim_utils import read_sim_data


def compare_correlations(r1, r2, n1, n2):
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se_diff_r = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    diff = z1 - z2
    z = abs(diff / se_diff_r)
    p = (1 - stats.norm.cdf(z)) * 2
    return z, p



r1, r2 = 0.8, 0.7
n1, n2 = 100, 80
z, p = compare_correlations(r1, r2, n1, n2)

warnings.filterwarnings("ignore")


plt.rcParams.update({
    'font.size': 10,
    "font.family": "sans-serif",
    'legend.frameon': False,
    'legend.fontsize': 9,
    'legend.handlelength': 1,
    'legend.title_fontsize': 0,
})
fn_prefix = 'fn'
title_padding = 20


def visualize_rt(fig_dir: Path, data_groups,
                 sim_df_all_prepotent,
                 sim_df_all_balanced,
                 sim_dt,
                 exp_dt,
                 ):
    plt.close('all')
    size = 3
    fig, axes = plt.subplots(
        2, 2, figsize=(size * 2, size * 2+1),
        sharex=True,
        sharey=True
    )

    order = ['AX', 'AY', 'BX', 'BY']

    pairs = [("AY", "AX"), ("AY", "BX"), ("AY", "BY")]

    rt_result = pd.concat([data_groups[ng]['bal'][0][['cue_probe', 'rt']]
                           for ng in data_groups.keys()], axis=0)
    rt_result['rt'] = rt_result['rt'] * exp_dt
    sns.boxplot(x="cue_probe", y="rt", data=rt_result,
                order=order, ax=axes[0, 0],
                color=color_pal_trial_types['Balanced'])
    Annotator(axes[0, 0], pairs, data=rt_result,
              x="cue_probe", y='rt', order=order
              ).configure(
        test='t-test_welch', text_format='star', loc='inside'
    ).apply_and_annotate()

    rt_result = pd.concat([data_groups[ng]['pre'][0][['cue_probe', 'rt']]
                           for ng in data_groups.keys()], axis=0)
    rt_result['rt'] = rt_result['rt'] * exp_dt

    sns.boxplot(x="cue_probe", y="rt", data=rt_result,
                order=order, ax=axes[0, 1],
                color=color_pal_trial_types['Prepotent'])
    Annotator(axes[0, 1], pairs, data=rt_result,
              x="cue_probe", y='rt', order=order
              ).configure(
        test='t-test_welch', text_format='star', loc='inside'
    ).apply_and_annotate()

    sim_df_all_balanced['rt'] = sim_df_all_balanced['rt'] * sim_dt
    sns.boxplot(x="cue_probe", y="rt", data=sim_df_all_balanced,
                order=order, ax=axes[1, 0],
                color=color_pal_trial_types['Balanced']
                )
    Annotator(axes[1, 0], pairs, data=sim_df_all_balanced,
              x="cue_probe", y='rt', order=order
              ).configure(
        test='t-test_welch', text_format='star', loc='inside'
    ).apply_and_annotate()

    sim_df_all_prepotent['rt'] = sim_df_all_prepotent['rt'] * sim_dt

    sns.boxplot(x="cue_probe", y="rt", data=sim_df_all_prepotent,
                order=order, ax=axes[1, 1],
                color=color_pal_trial_types['Prepotent']
                )

    Annotator(axes[1, 1], pairs, data=sim_df_all_prepotent,
              x="cue_probe", y='rt', order=order
              ).configure(
        test='t-test_welch', text_format='star', loc='inside'
    ).apply_and_annotate()
    for ax in axes.flatten():
        ax.set_xlabel('Trial type')
        ax.set_ylabel('')

    axes[0, 0].set_ylabel('Response time (ms)')
    axes[1, 0].set_ylabel('Response time (ms)')

    axes[0, 0].set_title('Balanced\n(Experiment)')
    axes[0, 1].set_title('Prepotent\n(Experiment)')
    axes[1, 0].set_title('Balanced\n(SPEL model simulation)')
    axes[1, 1].set_title('Prepotent\n(SPEL model simulation)')

    sns.despine()
    fig.tight_layout()
    for fm in ['svg', 'pdf', 'png']:
        specific_fig_dir = fig_dir / fm
        specific_fig_dir.mkdir(exist_ok=True)
        fig.savefig(specific_fig_dir /
                    f'fig-{fn_prefix}-RT-SC.{fm}', format=fm)


def count_selectives(A, B, C):
    mixed_selective = set(A) & set(
        B) & set(C)
    single_selective = set(A).difference(set(B), set(C)) | set(B).difference(
        set(A), set(C)) | set(C).difference(set(A), set(B))
    return len(mixed_selective), len(single_selective)


def visualize_venn(
    fig_dir: Path,

    n_ids_cue_correlated_all,
    n_ids_probe_correlated_all,
    n_ids_resp_correlated_all,

    n_ids_cue_correlated,
    n_ids_probe_correlated,
    n_ids_resp_correlated,
):
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    alpha = 0.5
    names = ('Cue', 'Probe', 'Response')
    set_colors = ('lightblue', 'hotpink', 'lightslategray')

    venn3([set(n_ids_cue_correlated_all),
           set(n_ids_probe_correlated_all),
           set(n_ids_resp_correlated_all)],
          names,
          set_colors=set_colors,
          alpha=alpha,
          ax=axes[0]
          )

    venn3([set(n_ids_cue_correlated),
           set(n_ids_probe_correlated),
           set(n_ids_resp_correlated)],
          names,
          set_colors=set_colors,
          alpha=alpha,
          ax=axes[1],
          )
    m_exp, s_exp = count_selectives(n_ids_cue_correlated_all, n_ids_probe_correlated_all, n_ids_resp_correlated_all)
    m_sim, s_sim = count_selectives(n_ids_cue_correlated, n_ids_probe_correlated, n_ids_resp_correlated)
    obs = [[m_exp, s_exp], [m_sim, s_sim]]
    chi2_stat, p_val, dof, expected = chi2_contingency(obs)

    print(f"Chi-square Statistic(Venn): {chi2_stat}")
    print(f"P-value: {p_val}")

    axes[0].set_title('Experiment')
    axes[1].set_title('SPEL model simulation')
    fig.tight_layout()
    for fm in ['svg', 'pdf', 'png']:
        specific_fig_dir = fig_dir / fm
        specific_fig_dir.mkdir(exist_ok=True)
        fig.savefig(specific_fig_dir /
                    f'fig-{fn_prefix}-Venn-SC.{fm}', format=fm)


def visualize_firing_rate(
    fig_dir: Path,

    collective_exp_hs_dict_mean_list_bal,
    collective_exp_hs_dict_mean_list_pre,
    nnid_idx_types_used,

    sim_hs_dict_all_balanced,
    sim_hs_dict_all_prepotent,
    sim_nids,

    sim_hs_dict_all_balanced_nr,
    sim_hs_dict_all_prepotent_nr,
    sim_nids_nr,

    sim_dt2phases,
    sim_dt,
    neuron_cutoff,
    extra=''
):
    size = 3
    plt.close('all')
    fig, axes = plt.subplots(3, 2, figsize=(
        size * 2 + 1, size * 2+1),
        sharey='row',
        sharex=True


    )

    ba_diff_bal_exp, ayx_diff_bal_exp = plot_neuron_frs_across_conditions(
        fig_dir, 'balanced',
        collective_exp_hs_dict_mean_list_bal,

        nnid_idx_types_used[1].copy(),
        sim_dt2phases,
        sim_dt,
        title='Experiment',
        ax=axes[0, 0],
        legend=False,
        returns_diff=True,
    )

    ba_diff_pre_exp, ayx_diff_pre_exp = plot_neuron_frs_across_conditions(
        fig_dir, 'prepontent',
        collective_exp_hs_dict_mean_list_pre,

        nnid_idx_types_used[1].copy(),
        sim_dt2phases,
        sim_dt,
        title='',
        ax=axes[0, 1],
        legend=True,
        returns_diff=True,
    )


    ba_diff_bal_sim, ayx_diff_bal_sim = plot_neuron_frs_across_conditions(
        fig_dir, 'balanced',
        sim_hs_dict_all_balanced,
        sim_nids[-neuron_cutoff:],
        sim_dt2phases,
        sim_dt,
        title='Simulation',
        ax=axes[1, 0],
        legend=False,
        returns_diff=True,
    )
    ba_diff_pre_sim, ayx_diff_pre_sim = plot_neuron_frs_across_conditions(
        fig_dir, 'prepontent',
        sim_hs_dict_all_prepotent,
        sim_nids[-neuron_cutoff:],
        sim_dt2phases,
        sim_dt,
        title='',
        ax=axes[1, 1],
        legend=True,
        returns_diff=True,
    )

    ba_diff_bal_sim, ayx_diff_bal_sim = plot_neuron_frs_across_conditions(
        fig_dir, 'balanced',
        sim_hs_dict_all_balanced_nr,
        sim_nids_nr[-neuron_cutoff:],
        sim_dt2phases,
        sim_dt,
        title='Simulation',
        ax=axes[2, 0],
        legend=False,
        returns_diff=True,
    )
    ba_diff_pre_sim, ayx_diff_pre_sim = plot_neuron_frs_across_conditions(
        fig_dir, 'prepontent',
        sim_hs_dict_all_prepotent_nr,
        sim_nids_nr[-neuron_cutoff:],
        sim_dt2phases,
        sim_dt,
        title='',
        ax=axes[2, 1],
        legend=True,
        returns_diff=True,
    )
    for ax in axes.flatten():
        ax.set_xlim(0, 4000)
        ax.set_ylabel('')

    axes[0, 0].set_ylim([0.1, 0.45])
    axes[1, 0].set_ylim([-0.05, 0.25])
    axes[2, 0].set_ylim([-0.05, 0.25])

    axes[0, 0].set_title('Balanced\n(Experiment)')
    axes[0, 1].set_title('Prepotent\n(Experiment)')

    axes[1, 0].set_title('Balanced\n(Simulation - SPEL model)')
    axes[1, 1].set_title('Prepotent\n(Simulation - SPEL model)')

    axes[2, 0].set_title('Balanced\n(Simulation - Classical RNN)')
    axes[2, 1].set_title('Prepotent\n(Simulation - Classical RNN)')

    axes[0, 0].set_ylabel('Firing rate\n(normalized)')
    axes[1, 0].set_ylabel('Firing rate\n(normalized)')
    axes[2, 0].set_ylabel('Firing rate\n(normalized)')

    sns.despine()
    fig.tight_layout(w_pad=0.2, h_pad=0.1)

    for fm in ['svg', 'pdf', 'png']:
        specific_fig_dir = fig_dir / fm
        specific_fig_dir.mkdir(exist_ok=True)
        fig.savefig(specific_fig_dir /
                    f'fig-{fn_prefix}{extra}-Switch-SC.{fm}', format=fm)


def visualize_decoding_accuracy(
    fig_dir: Path,
    exp_time_scores_a,
    exp_time_scores_b,
    sim_time_scores_a,
    sim_time_scores_b,
    sim_dt2phases_balanced,
    sim_dt,
):
    color_A = 'hotpink'
    color_B = 'lightslategray'
    lw = 3
    size = 3

    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(
        size * 2 + 1, size * 2), sharey='row', sharex=True)


    axes[0, 0].plot(
        exp_time_scores_a[:, 0] * sim_dt,
        exp_time_scores_a[:, 1],
        c=color_A,
        lw=lw,
    )
    axes[0, 1].plot(
        exp_time_scores_a[:, 0] * sim_dt,
        exp_time_scores_a[:, 1],
        ls='--',
        c=color_A,
        lw=lw,
    )
    axes[0, 1].plot(
        exp_time_scores_b[:, 0] * sim_dt,
        exp_time_scores_b[:, 1],
        c=color_B,
        lw=lw,
    )


    axes[1, 0].plot(
        sim_time_scores_a[:, 0] * sim_dt,
        sim_time_scores_a[:, 1],
        c=color_A,
        lw=lw,
    )

    axes[1, 1].plot(
        sim_time_scores_a[:, 0] * sim_dt,
        sim_time_scores_a[:, 1],
        ls='--',
        c=color_A,
        lw=lw,
    )
    axes[1, 1].plot(
        sim_time_scores_b[:, 0] * sim_dt,
        sim_time_scores_b[:, 1],
        c=color_B,
        lw=lw,
    )

    axes[1, 0].set_title('A-cue Trials\n(SPEL model simulation)')
    axes[1, 1].set_title('B-cue Trials\n(SPEL model simulation)')
    axes[0, 0].set_title('A-cue Trials\n(Experiment)')
    axes[0, 1].set_title('B-cue Trials\n(Experiment)')

    axes[0, 0].set_ylabel('Decoding accuracy')
    axes[1, 0].set_ylabel('Decoding accuracy')

    for ax in axes.flatten():
        ax.axvspan(sim_dt2phases_balanced['cue'][0] * sim_dt,
                   sim_dt2phases_balanced['cue'][1] * sim_dt,
                   color='silver',
                   alpha=0.2)
        ax.axvspan(sim_dt2phases_balanced['probe'][0] * sim_dt,
                   sim_dt2phases_balanced['probe'][1] * sim_dt,
                   color='silver',
                   alpha=0.2)
        ax.axhline(0.5, ls='dotted', color='black')
        ax.set_xlabel('Time/ms')

    dt2phases = dt2phase_ids[int(sim_dt)]
    for ax in axes.flatten():
        ax.set_xlim([0, 4000])
        ax.spines[['right', 'top']].set_visible(False)

        ax2 = ax.secondary_xaxis('top')
        ax2.set_xticks([
            (dt2phases['cue'][0] * sim_dt + dt2phases[
                'cue'][1] * sim_dt) / 2,
            (dt2phases['probe'][0] * sim_dt +
             dt2phases['probe'][1] * sim_dt) / 2
        ],
            minor=False)
        ax2.set_xticklabels(['cue', 'probe'])
        xticklabels = ax2.get_xticklabels()
        for label in xticklabels:
            label.set_color('gray')
        ax2.tick_params(axis='both', length=0)
        ax2.spines[['right', 'top']].set_visible(False)

    sns.despine()
    fig.tight_layout()
    for fm in ['svg', 'pdf', 'png']:
        specific_fig_dir = fig_dir / fm
        specific_fig_dir.mkdir(exist_ok=True)
        fig.savefig(
            specific_fig_dir / f'fig-{fn_prefix}-DecodingAccuracy-SC.{fm}', format=fm,)


def dynamics2df(pre_dyn, trial_type):
    train_prepotent_df = pd.DataFrame(
        read_train_dynamics(pre_dyn),
        columns=['Epoch', 'Loss', 'Correct rate'])
    train_prepotent_df['Trial'] = trial_type
    return train_prepotent_df


def read_training_dynamics(pre_dyn, bal_dyn):
    train_prepotent_df = dynamics2df(pre_dyn, 'Prepotent')
    train_balanced_df = dynamics2df(bal_dyn, 'Balanced')
    prepotent_end_ep = train_prepotent_df['Epoch'].max()
    train_balanced_df['Epoch'] += prepotent_end_ep - 1
    balance_end_ep = train_balanced_df['Epoch'].max()
    train_df = pd.concat(
        [train_prepotent_df,
         train_balanced_df], axis=0
    )
    train_df = train_df.reset_index()
    return train_df, prepotent_end_ep, balance_end_ep


def visualize_train_dynamics(
    fig_dir: Path,
    pre_dyn,
    bal_dyn,

    pre_dyn_esn,
    bal_dyn_esn,

    pre_dyn_nrnn,
    bal_dyn_nrnn,
):

    train_df, prepotent_end_ep, balance_end_ep = read_training_dynamics(
        pre_dyn, bal_dyn)



    train_df_esn, _, _ = read_training_dynamics(pre_dyn_esn, bal_dyn_esn)

    train_df_nrnn, _, _ = read_training_dynamics(pre_dyn_nrnn, bal_dyn_nrnn)

    c_pal = {
        'Prepotent': 'brown',
        'Balanced': 'steelblue'
    }

    c_th = 90
    size = 3
    sns.set_style('white',
                  rc={
                      'xtick.bottom': True,
                      'ytick.left': True,
                  },
                  )
    fig, axes = plt.subplots(2, 3,
                             sharex=True,
                             sharey='row',
                             figsize=(size * 2 + 1, size*1.5),
                             )
    alpha = 0.05
    lw = 3
    for ax in axes.flatten():
        ax.axvspan(1, prepotent_end_ep,
                   color=c_pal['Prepotent'],
                   alpha=alpha)
        ax.axvspan(prepotent_end_ep,
                   balance_end_ep,
                   color=c_pal['Balanced'],
                   alpha=alpha)
        ax.set_xlim(1, balance_end_ep)

    axes[0, 0].set_title('SPEL Model')
    sns.lineplot(x="Epoch", y="Loss",
                 hue="Trial",
                 data=train_df,
                 ax=axes[0, 0],
                 legend=False,
                 palette=c_pal,
                 linewidth=lw,
                 )

    axes[1, 0].axhline(c_th, color='gray', linestyle='dotted')
    sns.lineplot(x="Epoch", y="Correct rate",
                 hue="Trial",
                 data=train_df,
                 ax=axes[1, 0],
                 palette=c_pal,
                 linewidth=lw,
                 legend='brief'
                 )
    axes[1, 0].set_ylabel('Correct rate (%)')

    axes[0, 1].set_title('Classical RNN')
    sns.lineplot(x="Epoch", y="Loss",
                 hue="Trial",
                 data=train_df_nrnn,
                 ax=axes[0, 1],
                 legend=False,
                 palette=c_pal,
                 linewidth=lw,
                 )
    axes[1, 1].axhline(c_th, color='gray', linestyle='dotted')
    sns.lineplot(x="Epoch", y="Correct rate",
                 hue="Trial",
                 data=train_df_nrnn,
                 ax=axes[1, 1],
                 palette=c_pal,
                 linewidth=lw,
                 legend='brief'
                 )


    axes[0, 2].set_title('Echo State Network')
    sns.lineplot(x="Epoch", y="Loss",
                 hue="Trial",
                 data=train_df_esn,
                 ax=axes[0, 2],
                 legend=False,
                 palette=c_pal,
                 linewidth=lw,
                 )
    axes[1, 2].axhline(c_th, color='gray', linestyle='dotted')
    sns.lineplot(x="Epoch", y="Correct rate",
                 hue="Trial",
                 data=train_df_esn,
                 ax=axes[1, 2],
                 palette=c_pal,
                 linewidth=lw,
                 legend='brief'
                 )


    sns.despine(fig)
    fig.tight_layout()

    fig.tight_layout()
    for fm in ['svg', 'pdf', 'png']:
        specific_fig_dir = fig_dir / fm
        specific_fig_dir.mkdir(exist_ok=True)
        fig.savefig(specific_fig_dir /
                    f'fig-{fn_prefix}-training-3.{fm}', format=fm)


def execute_analysis(
        sim_dir,
        dat_exp,
        dat_exp_dir,
        cache, write, seed,
):
    sim_dir = Path(sim_dir)
    assert sim_dir.exists()
    fig_dir = sim_dir / 'figs'
    fig_dir.mkdir(exist_ok=True)
    print(fig_dir)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    if cache:
        data4plotting = torch.load(sim_dir / '.data4plotting.pt')
        print('Loading cached data')
    else:
        neuron_cutoff = 22
        exp_dt = 50

        dataf = sio.loadmat(dat_exp)['statsOut'][:, [3, 17]].astype(int)
        exp_ntypes = np.unique(dataf[:, 1])
        nids_types = {ntype: dataf[dataf[:, 1] == ntype, 0]
                      for ntype in exp_ntypes}


        exp_data_dir = Path(dat_exp_dir)
        neuron_groups = OrderedDict()
        for p in exp_data_dir.glob('*.mat'):
            if 'dc-2' not in p.stem and 'dc-1' not in p.stem:
                continue

            title_info = p.stem.split('_')

            title_info[5]
            trial_type = title_info[3]
            neuron_group = '_'.join(title_info[4:])
            if neuron_group not in neuron_groups:
                neuron_groups[neuron_group] = dict()

            neuron_groups[neuron_group][trial_type] = p

        exp_data_groups = OrderedDict()
        nnids_shareds = []
        df_all_pre_all = []
        for ng_idx, (ng, ng_f) in enumerate(neuron_groups.items()):
            if ng not in exp_data_groups:
                exp_data_groups[ng] = dict()

            assert 'pre' in ng_f and 'bal' in ng_f

            try:
                df_all_pre, hs_dict_all_pre, nnids_pre = preprocess_dataf(
                    ng_f['pre'])
                df_all_bal, hs_dict_all_bal, nnids_bal = preprocess_dataf(
                    ng_f['bal'])
            except Exception as e:
                print(e)
                continue

            nnids_shared = np.intersect1d(nnids_pre, nnids_bal)
            df_all_pre, hs_dict_all_pre = filter_data(
                df_all_pre, hs_dict_all_pre, nnids_pre, nnids_shared)
            df_all_bal, hs_dict_all_bal = filter_data(
                df_all_bal, hs_dict_all_bal, nnids_bal, nnids_shared)
            df_all_pre['ng_idx'] = ng_idx
            df_all_pre_all.append(df_all_pre.copy())


            exp_data_groups[ng]['pre'] = (
                df_all_pre, hs_dict_all_pre, nnids_shared)
            exp_data_groups[ng]['bal'] = (
                df_all_bal, hs_dict_all_bal, nnids_shared)
            exp_data_groups[ng]['nnids'] = nnids_shared

            nnids_shareds.append(nnids_shared.tolist())
            pass


        (
            (sim_trial_type_balanced,
             sim_dt_balanced,
             sim_dt2phases,
             sim_nids_balanced,
             sim_df_all_balanced,
             sim_hs_all_concat_balanced,
             sim_hs_dict_all_balanced),
            (sim_trial_type_prepotent,
             sim_dt_prepotent,
             sim_dt2phases_prepotent,
             sim_nids_prepotent,
             sim_df_all_prepotent,
             sim_hs_all_concat_prepotent,
             sim_hs_dict_all_prepotent),
            sim_dt,
            sim_nids,
        ) = read_sim_data(sim_dir)

        assert sim_dt2phases_prepotent == sim_dt2phases
        sim_df_all_prepotent = sim_df_all_prepotent.loc[
            sim_df_all_prepotent['rew'] == 1].copy()
        sim_df_all_balanced = sim_df_all_balanced.loc[
            sim_df_all_balanced['rew'] == 1].copy()

        (
            (sim_trial_type_balanced_nr,
             sim_dt_balanced_nr,
             sim_dt2phases_nr,
             sim_nids_balanced_nr,
             sim_df_all_balanced_nr,
             sim_hs_all_concat_balanced_nr,
             sim_hs_dict_all_balanced_nr),
            (sim_trial_type_prepotent_nr,
             sim_dt_prepotent_nr,
             sim_dt2phases_prepotent_nr,
             sim_nids_prepotent_nr,
             sim_df_all_prepotent_nr,
             sim_hs_all_concat_prepotent_nr,
             sim_hs_dict_all_prepotent_nr),
            sim_dt_nr,
            sim_nids_nr,
        ) = read_sim_data(sim_dir, fn_prefix='rnn_agent_records_normal_rnn')
        sim_df_all_prepotent_nr = sim_df_all_prepotent_nr.loc[
            sim_df_all_prepotent_nr['rew'] == 1].copy()
        sim_df_all_balanced_nr = sim_df_all_balanced_nr.loc[
            sim_df_all_balanced_nr['rew'] == 1].copy()

        print('Analyzing Venn diagram')
        exp_p_th = 0.05
        n_ids_cue_correlated_all = []
        n_ids_probe_correlated_all = []
        n_ids_resp_correlated_all = []

        n_ids_cue_correlated_all_dict = dict()
        n_ids_probe_correlated_all_dict = dict()
        n_ids_resp_correlated_all_dict = dict()

        n_ids_cue_correlated_all_plain = []
        n_ids_probe_correlated_all_plain = []
        n_ids_resp_correlated_all_plain = []

        n_id_offset = 0
        for ng_idx, ng in enumerate(exp_data_groups.keys()):
            nnids = exp_data_groups[ng]['nnids']

            df_all_types = pd.concat(
                [exp_data_groups[ng]['pre'][0], exp_data_groups[ng]['bal'][0]], axis=0)
            (n_ids_cue_correlated,
             n_ids_probe_correlated,
             n_ids_resp_correlated,
             ) = get_neuron_ids4encoding(
                nnids, df_all_types, p_th=exp_p_th)

            n_ids_cue_correlated_all_dict[ng] = n_ids_cue_correlated
            n_ids_probe_correlated_all_dict[ng] = n_ids_probe_correlated
            n_ids_resp_correlated_all_dict[ng] = n_ids_resp_correlated

            n_ids_cue_correlated_all_plain.append(
                np.array(n_ids_cue_correlated) + n_id_offset)
            n_ids_probe_correlated_all_plain.append(
                np.array(n_ids_probe_correlated) + n_id_offset)
            n_ids_resp_correlated_all_plain.append(
                np.array(n_ids_resp_correlated) + n_id_offset)

            n_ids_cue_correlated_all.extend(
                [f'g-{ng_idx}-{nid}' for nid in n_ids_cue_correlated]
            )
            n_ids_probe_correlated_all.extend(
                [f'g-{ng_idx}-{nid}' for nid in n_ids_probe_correlated]
            )
            n_ids_resp_correlated_all.extend(
                [f'g-{ng_idx}-{nid}' for nid in n_ids_resp_correlated]
            )
            n_id_offset += len(nnids)


        n_ids_cue_correlated_all_plain = np.concatenate(
            n_ids_cue_correlated_all_plain)
        n_ids_probe_correlated_all_plain = np.concatenate(
            n_ids_probe_correlated_all_plain)
        n_ids_resp_correlated_all_plain = np.concatenate(
            n_ids_resp_correlated_all_plain)
        nnid_idx_types_used = defaultdict(list)
        n_id_offset = 0
        for ng_idx, ng in enumerate(exp_data_groups.keys()):
            nnids = exp_data_groups[ng]['nnids']

            for ntype in nids_types.keys():
                nids_common = np.intersect1d(
                    nnids, nids_types[ntype])
                if len(nids_common) > 0:
                    nnid_idx_types_used[ntype] += (
                        np.where(np.in1d(nnids, nids_common))[
                            0] + n_id_offset).tolist()
                pass

            n_id_offset += len(nnids)
        nnid_idx_types_used = dict(nnid_idx_types_used)
        for ntype in nnid_idx_types_used.keys():
            nnid_idx_types_used[ntype] = np.array(nnid_idx_types_used[ntype])

        df_all = pd.concat([sim_df_all_balanced, sim_df_all_prepotent], axis=0)
        (n_ids_cue_correlated,
         n_ids_probe_correlated,
         n_ids_resp_correlated) = get_neuron_ids4encoding(sim_nids, df_all)


        (
            collective_exp_hs_dict_mean_list_pre,
            collective_exp_hs_dict_mean_list_bal,
            collective_exp_hs_dict_raw_lists_pre,
            collective_exp_hs_dict_raw_lists_bal,
            nnids_all
        ) = get_collective_hs(exp_data_groups)

        n_ids_cue_correlated_all_indices = []
        for nid in n_ids_cue_correlated_all_plain:
            nwhere = np.argwhere(nnids_all == nid)
            if len(nwhere) > 0:
                n_ids_cue_correlated_all_indices.append(nwhere[0][0])
        n_ids_cue_correlated_all_indices = np.array(
            n_ids_cue_correlated_all_indices)
        print('locations for cue: ', n_ids_cue_correlated_all_indices)

        n_ids_probe_correlated_all_indices = []
        for nid in n_ids_probe_correlated_all_plain:
            nwhere = np.argwhere(nnids_all == nid)
            if len(nwhere) > 0:
                n_ids_probe_correlated_all_indices.append(nwhere[0][0])
        n_ids_probe_correlated_all_indices = np.array(
            n_ids_probe_correlated_all_indices)
        print('locations for probe: ', n_ids_probe_correlated_all_indices)

        n_ids_resp_correlated_all_indices = []
        for nid in n_ids_resp_correlated_all_plain:
            nwhere = np.argwhere(nnids_all == nid)
            if len(nwhere) > 0:
                n_ids_resp_correlated_all_indices.append(nwhere[0][0])
        n_ids_resp_correlated_all_indices = np.array(
            n_ids_resp_correlated_all_indices)
        print('locations for response: ', n_ids_resp_correlated_all_indices)

        shared_indices = functools.reduce(
            np.intersect1d,
            (n_ids_cue_correlated_all_indices,
             n_ids_probe_correlated_all_indices,
             n_ids_resp_correlated_all_indices)
        )
        print('locations for all (together): ', shared_indices)

        print('Analyzing decoding accuracy')
        time_scores_a_list = []
        time_scores_b_list = []
        for g_name in exp_data_groups.keys():
            _, hs_dict_all_tmp, _ = exp_data_groups[g_name]['bal']

            try:
                time_scores_a, time_scores_b = get_AB_decoding_accuracy(
                    hs_dict_all_tmp, reduce=True)

                time_scores_a_list.append(time_scores_a)
                time_scores_b_list.append(time_scores_b)
            except Exception:
                pass
        exp_time_scores_a = np.stack(time_scores_a_list).mean(axis=0)
        exp_time_scores_b = np.stack(time_scores_b_list).mean(axis=0)
        sim_time_scores_a, sim_time_scores_b = get_AB_decoding_accuracy(
            sim_hs_dict_all_prepotent,
            probe_nnid_indices=sim_nids[:-neuron_cutoff],
        )

        data4plotting = {
            'sim_dt': sim_dt,
            'exp_dt': exp_dt,
            'fig_dir': fig_dir,
            'neuron_cutoff': neuron_cutoff,

            'data_groups': exp_data_groups,
            'sim_df_all_prepotent': sim_df_all_prepotent,
            'sim_df_all_balanced': sim_df_all_balanced,

            'n_ids_cue_correlated_all': n_ids_cue_correlated_all,
            'n_ids_probe_correlated_all': n_ids_probe_correlated_all,
            'n_ids_resp_correlated_all': n_ids_resp_correlated_all,
            'n_ids_cue_correlated': n_ids_cue_correlated,
            'n_ids_probe_correlated': n_ids_probe_correlated,
            'n_ids_resp_correlated': n_ids_resp_correlated,

            'collective_exp_hs_dict_mean_list_bal': \
            collective_exp_hs_dict_mean_list_bal,
            'collective_exp_hs_dict_mean_list_pre': \
            collective_exp_hs_dict_mean_list_pre,
            'nnid_idx_types_used': nnid_idx_types_used,
            'sim_hs_dict_all_balanced': sim_hs_dict_all_balanced,
            'sim_hs_dict_all_prepotent': sim_hs_dict_all_prepotent,
            'sim_nids': sim_nids,


            'sim_hs_dict_all_balanced_nr': sim_hs_dict_all_balanced_nr,
            'sim_hs_dict_all_prepotent_nr': sim_hs_dict_all_prepotent_nr,
            'sim_nids_nr': sim_nids_nr,

            'sim_dt2phases': sim_dt2phases,

            'exp_time_scores_a': exp_time_scores_a,
            'exp_time_scores_b': exp_time_scores_b,
            'sim_time_scores_a': sim_time_scores_a,
            'sim_time_scores_b': sim_time_scores_b,

            'df_all_pre_all': df_all_pre_all,
        }
        if write:
            print('writing to cache')
            torch.save(
                data4plotting, sim_dir / '.data4plotting.pt'
            )

    print('Creating figures')

    print('Visualize training dynamics')
    visualize_train_dynamics(
        fig_dir,
        sim_dir / 'correct_rates-prepotent.pth',
        sim_dir / 'correct_rates-balanced.pth',
        sim_dir / 'correct_rates_esn-prepotent.pth',
        sim_dir / 'correct_rates_esn-balanced.pth',
        sim_dir / 'correct_rates_esn-prepotent.pth',
        sim_dir / 'correct_rates_esn-balanced.pth',
    )
    print('Visualize RT')
    visualize_rt(
        fig_dir,
        data4plotting['data_groups'],
        data4plotting['sim_df_all_prepotent'].copy(),
        data4plotting['sim_df_all_balanced'].copy(),
        data4plotting['sim_dt'],
        data4plotting['exp_dt'],
    )
    print('Visualize venn')
    visualize_venn(
        fig_dir,

        data4plotting['n_ids_cue_correlated_all'],
        data4plotting['n_ids_probe_correlated_all'],
        data4plotting['n_ids_resp_correlated_all'],

        data4plotting['n_ids_cue_correlated'],
        data4plotting['n_ids_probe_correlated'],
        data4plotting['n_ids_resp_correlated'],
    )
    print('Visualize firing rate')
    visualize_firing_rate(
        fig_dir,

        data4plotting['collective_exp_hs_dict_mean_list_bal'],
        data4plotting['collective_exp_hs_dict_mean_list_pre'],
        data4plotting['nnid_idx_types_used'],

        data4plotting['sim_hs_dict_all_balanced'],
        data4plotting['sim_hs_dict_all_prepotent'],
        data4plotting['sim_nids'],


        data4plotting['sim_hs_dict_all_balanced_nr'],
        data4plotting['sim_hs_dict_all_prepotent_nr'],
        data4plotting['sim_nids_nr'],

        data4plotting['sim_dt2phases'],
        data4plotting['sim_dt'],
        data4plotting['neuron_cutoff'],
    )

    print('Visualize decoding accuracy')
    visualize_decoding_accuracy(
        fig_dir,
        data4plotting['exp_time_scores_a'],
        data4plotting['exp_time_scores_b'],
        data4plotting['sim_time_scores_a'],
        data4plotting['sim_time_scores_b'],
        data4plotting['sim_dt2phases'],
        data4plotting['sim_dt'],
    )


@click.command()
@click.option(
    '--sim-dir',
    default='data/sim/rnn_agent_records-balanced-gx6-100-20231003_013602.pth',
    help='Path to the simulation data',
    type=str)
@click.option(
    '--dat-exp',
    default='data/exp_new_w_nids/DPX_statsOut.mat',
    help='Path to the monkey data',
)
@click.option(
    '--dat-exp-dir',
    default='data/exp_new_w_nids/monkey_data',
    help='Path to the monkey data',
)
@click.option(
    '--cache',
    default=False,
    help='Use cached data or not',
)
@click.option(
    '--write',
    default=False,
    help='Write to cache if True',
)
@click.option('--seed', default=None, type=int)
def cli(
        sim_dir,
        dat_exp,
        dat_exp_dir,
        cache, write, seed
):
    print(dat_exp)
    execute_analysis(sim_dir,
                     dat_exp,
                     dat_exp_dir,
                     cache,
                     write,
                     seed)


if __name__ == '__main__':
    cli()
