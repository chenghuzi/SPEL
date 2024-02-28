from pathlib import Path
import scipy.io as sio
from collections import defaultdict
import numpy as np
import pandas as pd
import copy
import functools


cue_ids = {'A': 0, 'B': 1}
probe_ids = {'X': 0, 'Y': 1}

dt2phase_ids = {
    50: {
        'fix': (0, 10),
        'cue': (10, 30),
        'delay': (30, 50),
        'probe': (50, 60),
    },
    100: {
        'fix': (0, 5),
        'cue': (5, 15),
        'delay': (15, 25),
        'probe': (25, 30),
    },
    250: {
        'fix': (0, 2),
        'cue': (2, 6),
        'delay': (6, 10),
        'probe': (10, 12),
    }
}

trial_info_dict = {
    1: "AX",
    2: "AY",
    3: "BX",
    4: "BY",
}

dt = 50

sensory_mapping = {
    0: "no",
    1: "A",
    2: "B",
    3: "X",
    4: "Y",
}

dt2phases = dt2phase_ids[int(dt)]


def parse_trial_data(episode_dat, neuron_ids: list, all_neuron_ids: list):
    """Parse the episode data cell.

    Args:
        episode_dat (np.ndarray): A time frame matrix whose
            rows are differnet time slices and columns are different variables.
        neuron_ids: the ids of neurons in this trial.
        all_neuron_ids: the ids of neurons we want to keep.
    Returns:
        tuple: (cue, probe, hs, info_type, row).
            cue (str): cue type.
            probe (str): probe type.
            hs (np.ndarray): neural activity matrix.
            info_type (str): trial type.
            row: a list of features including cue, probe, target, rt, and
                mean activity of all neurons in each phase.
    """
    info_type = episode_dat[:, 0]
    assert len(np.unique(info_type)) == 1
    info_type = trial_info_dict[info_type[0]]
    if info_type == 'AX':
        target = 1
    else:
        target = 0
    cue, probe = list(info_type)
    cue_id = cue_ids[cue]
    probe_id = probe_ids[probe]

    timestamp = episode_dat[:, 1]

    start_idx = np.argwhere(timestamp >= -500)[:, 0][0]

    timestamp = timestamp[start_idx:]
    _sensory_input = episode_dat[start_idx:, 2]
    action = episode_dat[start_idx:, 3]

    hs = episode_dat[start_idx:, 4:]
    nid_indices = [neuron_ids.index(nid) for nid in all_neuron_ids]

    hs = hs[:, nid_indices]


    response_idx = np.argwhere(action != 0)[:, 0][0]
    rt = response_idx - dt2phases['probe'][0]


    beforecue_h_mean = hs[
        dt2phases['fix'][0]:dt2phases['fix'][1]].mean(axis=0).tolist()

    cue_h_mean = hs[
        dt2phases['cue'][0]:dt2phases['cue'][1]].mean(axis=0).tolist()


    delay_h_mean = hs[
        dt2phases['delay'][0]:dt2phases['delay'][1]].mean(axis=0).tolist()


    probe_h_mean = hs[
        dt2phases['probe'][0]:dt2phases['probe'][1]].mean(axis=0).tolist()

    ressponse_h_mean = hs[dt2phases['probe'][0]:-1].mean(axis=0).tolist()

    row = [info_type, cue_id, probe_id, target, rt] + beforecue_h_mean + \
        cue_h_mean + delay_h_mean + probe_h_mean + ressponse_h_mean
    return (
        cue,
        probe,
        hs,
        info_type,
        row
    )


def parse_dataf(d_f):
    _, nt = d_f['ens_trials'].shape
    assert _ == 1, \
        "parse data failed as the shape of ens_trials " + \
        f"is {d_f['ens_trials'].shape}"

    df_all = []
    trial_ids = []
    drug_conds = []
    hs_dict = defaultdict(list)
    neuron_idss = []
    for trial_idx in range(nt):
        trial_data = d_f['ens_trials'][0, trial_idx]
        neuron_ids = trial_data[0, 0][3].reshape(-1).tolist()
        neuron_idss.append(neuron_ids)

    all_neuron_ids = sorted(list(
        functools.reduce(lambda x, y: set(x) & set(y), neuron_idss)
    ))

    for trial_idx in range(nt):
        trial_data = d_f['ens_trials'][0, trial_idx]
        episode_data = trial_data[0, 0][0]
        rewarded = trial_data[0, 0][1][0, 0]
        _ens_num = trial_data[0, 0][2][0, 0]
        neuron_ids = trial_data[0, 0][3].reshape(-1).tolist()
        if rewarded == 0:
            continue

        trial_id = trial_data[0, 0][4].reshape(-1).item()
        drug_cond = trial_data[0, 0][5].reshape(-1).item()
        trial_ids.append(trial_id)
        drug_conds.append(drug_cond)

        (cue,
         probe,
         hs,
         info_type,
         row) = parse_trial_data(episode_data, neuron_ids, all_neuron_ids)

        hs_dict[(cue, probe)].append(hs)
        df_all.append([drug_cond, trial_id] + row)


    nids = np.array(all_neuron_ids)
    cols = [
        'drug_cond', 'trial_id', 'cue_probe', 'cue', 'probe', 'target', 'rt'] + [
        f'b_{idx}' for idx in nids] + [
            f'c_{idx}' for idx in nids] + [
        f'd_{idx}' for idx in nids] + [
            f'p_{idx}' for idx in nids] + [
                f'r_{idx}' for idx in nids]
    df_all = pd.DataFrame(df_all, columns=cols)
    df_all.sort_values(by=['trial_id'], ascending=True, inplace=True)
    df_all.drop(columns=['trial_id'], inplace=True)

    return df_all, hs_dict, nids


def filter_data(df_all, hs_dict_all, nnids, nnids_shared):
    filtered_cols = [
        'drug_cond', 'cue_probe', 'cue', 'probe', 'target', 'rt'] + [
        f'b_{idx}' for idx in nnids_shared] + [
            f'c_{idx}' for idx in nnids_shared] + [
        f'd_{idx}' for idx in nnids_shared] + [
            f'p_{idx}' for idx in nnids_shared] + [
                f'r_{idx}' for idx in nnids_shared]
    df_all = df_all[filtered_cols].copy()
    nnids = nnids.tolist()
    nnid_indices = [nnids.index(nid) for nid in nnids_shared]


    for k, v in hs_dict_all.items():
        hs_dict_all[k] = {
            'raw': v['raw'][:, :, nnid_indices],
            'mean': v['mean'][:, nnid_indices],
        }
    return df_all, hs_dict_all


def extract_data(
        data: dict,
        data_names: dict,
        monkey_groups: set,
        trial_type: str,
        verbose=True):
    df_all_list = []
    hs_dict_all_list = []
    nids_list = []

    for ng in monkey_groups:
        dat_fs = data[(ng, trial_type)]
        data_name = data_names[(ng, trial_type)]
        if verbose:
            dfns = '\n'.join(data_name)
            print(f'({trial_type}) data file names:\n{dfns}')
        num_nuerons_total = 0
        for block_idx, dat_f in enumerate(dat_fs):

            df_all, hs_dict, nnids = parse_dataf(dat_f)
            df_all['block_idx'] = block_idx

            num_nuerons = len(nnids)
            hs_dict_all = dict()
            for cue in cue_ids.keys():
                for probe in probe_ids.keys():
                    lengths = [hs.shape[0] for hs in hs_dict[(cue, probe)]]
                    nids = [hs.shape[1] for hs in hs_dict[(cue, probe)]]
                    assert len(np.unique(nids)) == 1
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

            num_nuerons_total += num_nuerons
            df_all_list.append(df_all.copy())
            hs_dict_all_list.append(copy.deepcopy(hs_dict_all))
            nids_list.append(nnids.copy())
        if verbose:
            print(f'monkey_group={ng}, num_nuerons={num_nuerons_total}')

    return df_all_list, hs_dict_all_list, nids_list


def read_exp_data(exp_data_dir: Path):
    data = defaultdict(list)
    data_names = defaultdict(list)
    neuron_groups = set()

    for p in exp_data_dir.glob('*.mat'):
        if 'dc-2' not in p.stem and 'dc-1' not in p.stem:
            continue
        d_info = p.stem.split('_')
        trial_type = d_info[3]
        neuron_group = d_info[5]
        neuron_groups = neuron_groups | {neuron_group}
        dataf = sio.loadmat(p)
        data[(neuron_group, trial_type)].append(dataf)
        data_names[(neuron_group, trial_type)].append(p.stem)

    neuron_groups = list(neuron_groups)
    print(f'monkey groups={neuron_groups}')

    df_all_list_pre, hs_dict_all_list_pre, nids_list_pre = extract_data(
        data,
        data_names,
        neuron_groups,
        'pre'
    )
    df_all_list_bal, hs_dict_all_list_bal, nids_list_bal = extract_data(
        data,
        data_names,
        neuron_groups,
        'bal'
    )
    return (
        (df_all_list_pre, hs_dict_all_list_pre, nids_list_pre),
        (df_all_list_bal, hs_dict_all_list_bal, nids_list_bal),
        dt,
        dt2phases,
    )
