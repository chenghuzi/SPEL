from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pingouin import ancova
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import scipy.io as sio
from scipy.stats import pearsonr
from exp_utils import (
    parse_dataf,
    cue_ids,
    probe_ids,
    dt2phase_ids,
)
from sklearn.model_selection import cross_val_score
from skimage.measure import block_reduce
from seq_pred_machine import encode_inputs

epislon = 1e-22


color_pal_cueprobes = {
    "AX": "#F7AD57",
    "AY": "#F1565C",
    "BX": "#AA5DA5",
    "BY": "#5598CA",
}

color_pal_trial_types = {
    "Prepotent": "#63719c",
    "Balanced": "#bc8671",
}


class BinaryLinearRegression(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.lr = LinearRegression()

    def fit(self, X, y):
        self.lr.fit(X, y)
        return self

    def predict(self, X):
        return [1 if x else 0 for x in self.lr.predict(X) > 0.5]

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


def preprocess(fn, dt, use_e=True):
    fn = Path(fn)
    assert fn.exists(), f"{fn} does not exist"
    _, trial_type = fn.stem.split('-')
    records = torch.load(fn)

    print(f'samples ({trial_type}): ', len(records[trial_type]))

    g_idx = -1

    records[trial_type][0][0][0][g_idx]

    nids = np.arange(len(records[trial_type][0][0][0][-3]) +
                     len(records[trial_type][0][0][0][-1]))


    [f'n_{idx}' for idx in nids]
    df_all = []

    cols = ['rew', 'cue_probe', 'cue', 'probe', 'target', 'rt'] + [
        f'b_{idx}' for idx in nids] + [f'c_{idx}' for idx in nids] + [
        f'd_{idx}' for idx in nids] + [f'p_{idx}' for idx in nids] + [
            f'r_{idx}' for idx in nids]

    hs_dict = defaultdict(list)

    dt2phases = dt2phase_ids[int(dt)]

    for episode_extra in records[trial_type]:
        episode, rew = episode_extra

        if rew == 0:
            continue

        hs = []

        for i in range(len(episode)):
            obs, act, obs_next, reward, done, info, h, o, e = episode[i]
            hs.append(np.concatenate([h, e]))



        hs = np.array(hs)


        if info['type'] == 'AX':
            target = 1
        else:
            target = 0

        cue, probe = list(info['type'])
        cue_id = cue_ids[cue]
        probe_id = probe_ids[probe]

        beforecue_h_mean = hs[
            dt2phases['fix'][0]:dt2phases['fix'][1]].mean(axis=0).tolist()

        cue_h_mean = hs[
            dt2phases['cue'][0]:dt2phases['cue'][1]].mean(axis=0).tolist()


        delay_h_mean = hs[
            dt2phases['delay'][0]:dt2phases['delay'][1]].mean(axis=0).tolist()


        probe_h_mean = hs[
            dt2phases['probe'][0]:dt2phases['probe'][1]].mean(axis=0).tolist()

        ressponse_h_mean = hs[dt2phases['probe'][0]:-1].mean(axis=0).tolist()


        rt = hs.shape[0] - dt2phases['probe'][0]

        df_all.append([
            rew, info['type'], cue_id, probe_id, target, rt
        ] + beforecue_h_mean + cue_h_mean +
            delay_h_mean + probe_h_mean + ressponse_h_mean)


        hs_dict[(cue, probe)].append(hs)

    df_all = pd.DataFrame(df_all, columns=cols)


    hs_dict_all = dict()
    for cue in cue_ids.keys():
        for probe in probe_ids.keys():
            lengths = [hs.shape[0] for hs in hs_dict[(cue, probe)]]
            max_length = max(lengths)

            hsall = np.zeros(
                (len(hs_dict[(cue, probe)]), max_length, hs.shape[1]))
            for hs_idx, hs in enumerate(hs_dict[(cue, probe)]):
                hsall[hs_idx, 0:hs.shape[0], :] = hs


            hs_dict_all[(cue, probe)] = {
                'raw': hsall.copy(),
                'mean': hsall.mean(axis=0),
            }

    hs_all_concat = []
    for cue in cue_ids.keys():
        for probe in probe_ids.keys():
            hs_all_concat.append(hs_dict_all[(cue, probe)]['mean'])
    hs_all_concat = np.concatenate(hs_all_concat, axis=0)

    return trial_type, dt, dt2phases, nids, df_all, hs_all_concat, hs_dict_all


def get_neuron_ids4encoding(nids, df_all, p_th=0.0001):
    pv_probes = {}
    pv_cues = {}
    pv_resps = {}

    for neuron_id in nids:
        cue_res = ancova(
            data=df_all, dv=f'c_{neuron_id}',
            covar=[
                f'b_{neuron_id}',
                f'd_{neuron_id}',
                f'p_{neuron_id}'], between='cue')
        pv_cues[neuron_id] = cue_res['p-unc'][0]

        probe_res = ancova(
            data=df_all, dv=f'p_{neuron_id}',
            covar=[
                f'b_{neuron_id}',
                f'd_{neuron_id}',
                f'c_{neuron_id}'], between='probe')
        pv_probes[neuron_id] = probe_res['p-unc'][0]

        resp_res = ancova(
            data=df_all, dv=f'r_{neuron_id}',
            covar=[
                f'b_{neuron_id}',
                f'd_{neuron_id}',
                f'c_{neuron_id}'], between='target')
        pv_resps[neuron_id] = resp_res['p-unc'][0]

    n_ids_cue_correlated = [k for k, v in pv_cues.items() if v < p_th]
    n_ids_probe_correlated = [k for k, v in pv_probes.items() if v < p_th]
    n_ids_resp_correlated = [k for k, v in pv_resps.items() if v < p_th]

    return n_ids_cue_correlated, n_ids_probe_correlated, n_ids_resp_correlated


def plot_neuron_frs_across_conditions(fig_dir, trial_type, hs_dict_all,
                                      n_ids_cue_correlated,
                                      dt2phases, dt, title="", ax=None,
                                      legend=True,
                                      returns_diff=False, lw=3):
    if ax is None:
        plt.close('all')
        fig, ax = plt.subplots(figsize=[4, 4])

    if type(n_ids_cue_correlated) is not np.ndarray:
        n_ids_cue_correlated = np.array([n_ids_cue_correlated])

    bxc = hs_dict_all[('B', 'X')]['mean'][:, n_ids_cue_correlated]
    byc = hs_dict_all[('B', 'Y')]['mean'][:, n_ids_cue_correlated]
    axc = hs_dict_all[('A', 'X')]['mean'][:, n_ids_cue_correlated]
    ayc = hs_dict_all[('A', 'Y')]['mean'][:, n_ids_cue_correlated]


    ax.plot(np.arange(bxc.shape[0]) * dt, bxc.mean(axis=1),
            label='BX', c=color_pal_cueprobes['BX'], lw=lw)
    ax.plot(np.arange(byc.shape[0]) * dt, byc.mean(axis=1),
            label='BY', c=color_pal_cueprobes['BY'], lw=lw)
    ax.plot(np.arange(axc.shape[0]) * dt, axc.mean(axis=1),
            label='AX', c=color_pal_cueprobes['AX'], lw=lw)
    ax.plot(np.arange(ayc.shape[0]) * dt, ayc.mean(axis=1),
            label='AY', c=color_pal_cueprobes['AY'], lw=lw)

    ax.axvspan(dt2phases['cue'][0] * dt, dt2phases['cue'][1] * dt,
               color='silver',
               alpha=0.2)
    ax.axvspan(dt2phases['probe'][0] * dt, dt2phases['probe'][1] * dt,
               color='silver',
               alpha=0.2)
    ax.set_xlabel('Time/ms')
    if legend:
        ax.legend(fontsize=11, frameon=False,
                  labelcolor='linecolor',
                  handlelength=0,
                  loc='upper right'
                  )
    ax.set_ylabel('Fr (Normalized)')
    ax.spines[['right', 'top']].set_visible(False)

    ax2 = ax.secondary_xaxis('top')
    ax2.set_xticks([(dt2phases['cue'][0] * dt + dt2phases['cue'][1] * dt) / 2,
                    (dt2phases['probe'][0] * dt +
                     dt2phases['probe'][1] * dt) / 2
                    ],
                   minor=False)
    ax2.set_xticklabels(['cue', 'probe'])
    xticklabels = ax2.get_xticklabels()
    for label in xticklabels:
        label.set_color('gray')

    ax2.tick_params(axis='both', length=0)

    ax2.spines[['right', 'top']].set_visible(False)

    ax.set_title(title)
    if ax is None:
        plt.tight_layout()
        plt.savefig(fig_dir / f'CUE-S-{trial_type}.png')

    if returns_diff:
        min_length = min([bxc.shape[0],
                          byc.shape[0],
                          axc.shape[0],
                          ayc.shape[0]])
        ba_diff = bxc[:min_length, :] - axc[:min_length, :]
        ayx_diff = ayc[:min_length, :] - axc[:min_length, :]
        return ba_diff, ayx_diff


def read_train_dynamics(train_f):
    prepotent_train_d = torch.load(train_f)
    plain_loss = np.concatenate(prepotent_train_d['losss'])
    epoch_nums = np.concatenate([np.ones(len(loss)) * (idx + 1)
                                for idx, loss in enumerate(prepotent_train_d['losss'])])
    correct_rates = np.concatenate([
        np.ones(len(loss)) * prepotent_train_d['correct_rates'][idx] * 100
        for idx, loss in enumerate(prepotent_train_d['losss'])])
    data = np.stack([epoch_nums, plain_loss,
                    correct_rates], axis=0).T
    return data


def get_ws_target_ratio(df_tmp, window_size):
    return df_tmp['target'].rolling(
        window=window_size).apply(lambda x: np.mean(x[:-1]))


def get_history_effect(df_all, dt, window_sizes):
    df_all['cue_probe_num'] = df_all['cue'] * 2\
        + df_all['probe'] * 1
    df_all['rt'] = df_all['rt'] * dt

    assert max(window_sizes) < len(
        df_all) // 2, f'window_size_max should be smaller than {len(df_all)//2}'
    corr_info_all = []
    for ws_idx, window_size in enumerate(window_sizes):
        df_tmp = df_all.copy()
        col_history = f'history_target_ratio-{window_size}'
        df_tmp[col_history] = get_ws_target_ratio(df_tmp, window_size)
        df_tmp[col_history].corr(
            df_tmp['rt'])
        df_tmp.dropna(inplace=True)

        df_tmp_nonay = df_tmp.loc[df_tmp['cue_probe'] != 'AY'].copy()
        corr = pearsonr(df_tmp_nonay[col_history], df_tmp_nonay['rt'])
        corr_info_all.append(
            (corr, len(df_tmp_nonay[col_history]))
        )
    return corr_info_all


def extract_sim_history_effect(
    sim_df_all,
    window_sizes,
    sim_dt,
    n_permutations=5,
    sample_chunk_size=800,
):
    corr_info_pre_ordered_sim = []
    corr_info_pre_permuted_sim = []
    for _ in range(15):
        c_idx = np.random.randint(0, sim_df_all.shape[0] - sample_chunk_size)

        sim_df_chunked = sim_df_all.iloc[
            c_idx: c_idx + sample_chunk_size].copy()

        corr_info_pre_ordered_sim.append(get_history_effect(
            sim_df_chunked.copy(), sim_dt, window_sizes))

        for _ in range(n_permutations):
            corr_info_pre_permuted_sim.append(get_history_effect(
                sim_df_chunked.sample(frac=1).copy(), sim_dt, window_sizes))

    corr_info_pre_ordered_sim = list(
        map(list, zip(*corr_info_pre_ordered_sim)))

    corr_info_pre_permuted_sim = list(
        map(list, zip(*corr_info_pre_permuted_sim)))

    ws_history_effect_ordered_sim = {}
    ws_history_effect_permuted_sim = {}
    for ws, corr_ordered, corr_permuted in zip(
            window_sizes,
            corr_info_pre_ordered_sim,
            corr_info_pre_permuted_sim):

        r1s = np.array([rn[0].statistic for rn in corr_ordered])
        ws_history_effect_ordered_sim[ws] = r1s

        r2s = np.array([rn[0].statistic for rn in corr_permuted])
        ws_history_effect_permuted_sim[ws] = r2s

    return ws_history_effect_ordered_sim, ws_history_effect_permuted_sim


def preprocess_dataf(datf):
    df_all, hs_dict, nnids = parse_dataf(sio.loadmat(datf))

    hs_dict_all = dict()
    for cue in cue_ids.keys():
        for probe in probe_ids.keys():
            lengths = [hs.shape[0] for hs in hs_dict[(cue, probe)]]
            nids = [hs.shape[1] for hs in hs_dict[(cue, probe)]]
            err_info = f'In {datf}, Number of neurons \
                is not the same, they are {nids}'
            assert len(np.unique(nids)) == 1, err_info
            nids = nids[0]
            max_length = max(lengths)


            hsall = np.zeros(
                (len(hs_dict[(cue, probe)]), max_length, nids))
            for hs_idx, hs in enumerate(hs_dict[(cue, probe)]):
                hsall[hs_idx, 0:hs.shape[0], :] = hs

            hs_dict_all[(cue, probe)] = {
                'raw': hsall.copy(),
                'mean': hsall.mean(axis=0),
            }
    return df_all, hs_dict_all, nnids


def get_AB_decoding_accuracy(hh, probe_nnid_indices=None, reduce=False):

    axm = hh[('A', 'X')]['raw']
    aym = hh[('A', 'Y')]['raw']


    bxm = hh[('B', 'X')]['raw']
    bym = hh[('B', 'Y')]['raw']

    if probe_nnid_indices is not None:
        axm = axm[:, :, probe_nnid_indices]
        aym = aym[:, :, probe_nnid_indices]

        bxm = bxm[:, :, probe_nnid_indices]
        bym = bym[:, :, probe_nnid_indices]

    if reduce:
        axm = block_reduce(
            axm,
            block_size=(1, 2, 1),
            func=np.mean,
            cval=np.mean(axm))
        aym = block_reduce(
            aym,
            block_size=(1, 2, 1),
            func=np.mean,
            cval=np.mean(aym))

        bxm = block_reduce(
            bxm,
            block_size=(1, 2, 1),
            func=np.mean,
            cval=np.mean(bxm))
        bym = block_reduce(
            bym,
            block_size=(1, 2, 1),
            func=np.mean,
            cval=np.mean(bym))

    num_trials_a = min(axm.shape[0], aym.shape[0])
    axm = axm[:num_trials_a]
    aym = aym[:num_trials_a]


    num_trials_b = min(bxm.shape[0], bym.shape[0])
    bxm = bxm[:num_trials_b]
    bym = bym[:num_trials_b]

    time_scores_a = []
    for t_idx in range(min(axm.shape[1], aym.shape[1])):
        clf = BinaryLinearRegression()

        data = np.concatenate([axm[:, t_idx, :], aym[:, t_idx, :]], axis=0)
        labels = np.concatenate(
            [np.zeros(axm.shape[0]), np.ones(aym.shape[0])], axis=0)

        score = cross_val_score(clf, data, labels).mean()
        time_scores_a.append((t_idx, score))

    time_scores_b = []
    for t_idx in range(min(bxm.shape[1], bym.shape[1])):
        clf = BinaryLinearRegression()

        data = np.concatenate([bxm[:, t_idx, :], bym[:, t_idx, :]], axis=0)
        labels = np.concatenate(
            [np.zeros(bxm.shape[0]), np.ones(bym.shape[0])], axis=0)

        score = cross_val_score(clf, data, labels, cv=5).mean()
        time_scores_b.append((t_idx, score))

    return np.array(time_scores_a), np.array(time_scores_b)


def get_collective_hs(exp_data_groups):
    collective_exp_hs_dict_mean_list_pre = defaultdict(list)
    collective_exp_hs_dict_mean_list_bal = defaultdict(list)
    collective_exp_hs_dict_raw_lists_pre = defaultdict(list)
    collective_exp_hs_dict_raw_lists_bal = defaultdict(list)

    ng_names = list(exp_data_groups.keys())
    four_conditions = exp_data_groups[ng_names[0]]['pre'][1].keys()

    for cond in four_conditions:
        nnids_all = []
        for ng_name in ng_names:
            nnids_all.append(exp_data_groups[ng_name]['nnids'])
            collective_exp_hs_dict_mean_list_pre[cond].append(
                exp_data_groups[ng_name]['pre'][1][cond]['mean']
            )
            collective_exp_hs_dict_raw_lists_pre[cond].append(
                exp_data_groups[ng_name]['pre'][1][cond]['raw']
            )

            collective_exp_hs_dict_mean_list_bal[cond].append(
                exp_data_groups[ng_name]['bal'][1][cond]['mean']
            )
            collective_exp_hs_dict_raw_lists_bal[cond].append(
                exp_data_groups[ng_name]['bal'][1][cond]['raw']
            )

        nnids_all = np.concatenate(nnids_all)
        collective_exp_hs_dict_mean_list_bal[cond] = np.concatenate(
            collective_exp_hs_dict_mean_list_bal[cond],
            axis=1)

        collective_exp_hs_dict_mean_list_bal[cond] = {
            'mean': block_reduce(
                collective_exp_hs_dict_mean_list_bal[cond],
                block_size=(2, 1),
                func=np.mean,
                cval=np.mean(collective_exp_hs_dict_mean_list_bal[cond])),
        }
        collective_exp_hs_dict_mean_list_pre[cond] = np.concatenate(
            collective_exp_hs_dict_mean_list_pre[cond],
            axis=1)
        collective_exp_hs_dict_mean_list_pre[cond] = {
            'mean': block_reduce(
                collective_exp_hs_dict_mean_list_pre[cond],
                block_size=(2, 1),
                func=np.mean,
                cval=np.mean(collective_exp_hs_dict_mean_list_pre[cond]))
        }
    return (
        collective_exp_hs_dict_mean_list_pre,
        collective_exp_hs_dict_mean_list_bal,
        collective_exp_hs_dict_raw_lists_pre,
        collective_exp_hs_dict_raw_lists_bal,
        nnids_all
    )
