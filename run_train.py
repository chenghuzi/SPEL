import copy
import random
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import torch
from tqdm import tqdm

from analysis import dt2phase_ids
from run_analysis import execute_analysis
from seq_pred_machine import (
    DTYPE, AgentNN, compute_rep_loss, encode_inputs, device
)
from task import TRIAL_TYPES, AXBYEnv, gen_training_episodes


@contextmanager
def nullcontext(enter_result=None):
    yield enter_result


SAMPLE2TEST = 3000


def gen_eval_episodes(agent: AgentNN, num_episodes, trial_type,
                      dt=250, show_pbar=True, include_h=False,
                      inclure_rew_info=False, perform_BP=False):
    """
    Some prior assumptions are made here: the agent has already learned to stay
        there.
    and fixate in most of the time.
    """

    assert trial_type in TRIAL_TYPES
    env = AXBYEnv(trial_type, dt=dt)
    rewed_episodes_count = 0

    episodes = []
    episode = []

    def generator():
        while len(episodes) < num_episodes:
            yield

    obs, info = env.reset()
    act = 0
    reward = 0
    done = False

    h, o = agent.reset_h()

    if show_pbar:
        pbar = tqdm(generator())
    else:
        pbar = generator()
    if perform_BP:
        ctx = nullcontext()
    else:
        ctx = torch.no_grad()
    with ctx:
        if perform_BP:
            agent.optim.zero_grad()
            loss = 0
        for _ in pbar:
            act, h, o = agent.select_action(h,
                                            encode_inputs(
                                                obs, act, reward, done
                                            ),
                                            o.data.detach()
                                            )

            obs_next, reward, terminated, truncated, info = env.step(act)
            done = truncated or terminated

            y = encode_inputs(obs, act, reward, done)

            if agent.model_type == 'normal_rnn':
                input_signal = torch.abs(y)
            else:
                input_signal = torch.abs(o - y)

            if perform_BP:
                loss += compute_rep_loss(o, y)

            if include_h is False:
                episode.append(
                    (obs.copy(), act, obs_next.copy(), reward, done, info)
                )
            else:
                episode.append(
                    (obs.copy(), act, obs_next.copy(),
                        reward, done, info, h.data.cpu().numpy(),
                        o.data.cpu().numpy(),
                        input_signal.data.cpu().numpy())
                )
            obs = obs_next
            if done:
                if reward > 0:
                    rewed_episodes_count += 1

                    if perform_BP:
                        agent.optim.zero_grad()
                        loss = loss / (len(episode) + 1)
                        loss = loss + \
                            agent.w_penalty_weight * agent.compute_w_reg()
                        loss.backward()
                        agent.optim.step()

                        loss = 0
                else:
                    if perform_BP:
                        agent.optim.zero_grad()
                        loss = 0

                if inclure_rew_info:
                    episodes.append((copy.deepcopy(episode), reward))
                else:
                    episodes.append(copy.deepcopy(episode))

                episode = []
                pbar.set_postfix({'rewed': rewed_episodes_count,
                                  'total': len(episodes)})

                obs, info = env.reset()
                act = 0
                reward = 0
                done = False
                h, o = agent.reset_h()

    agent.optim.zero_grad()

    cr = rewed_episodes_count / len(episodes)
    return episodes, cr


@click.group()
@click.option('--seed', default=1999, type=int)
def cli(seed):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    pass



def gen_training_samples(trial_type, total_t_mature, dt=250):
    training_episode_reps = gen_training_episodes(
        total_t_mature, trial_type, p_f=1.0, p_j=0.99, dt=dt)
    rewarded = np.array([int(ep_rep[1]) for ep_rep in training_episode_reps])
    avg_lengths = np.array([len(ep_rep[0])
                           for ep_rep in training_episode_reps])
    print(
        f'training lengths: mean={np.mean(avg_lengths)}, '
        f'std={np.std(avg_lengths)}')
    rewarded_cr = np.mean(rewarded)

    indices_rewarded = np.argwhere(rewarded == 1)[:, 0]
    episode_reps_rewarded = [training_episode_reps[idx][0]
                             for idx in indices_rewarded]

    return episode_reps_rewarded, rewarded_cr


def train(agent: AgentNN,
          n_epochs, episode_reps_rewarded,
          stop_cr, dt, trial_type,
          samples_required=200, cr_stop=0.9, eval_every_ep=False,):
    sample2test = SAMPLE2TEST
    records_fn = f'rnn_agent_records-{trial_type}.pth'
    correct_rates_fn = f'correct_rates-{trial_type}.pth'
    if agent.model_type != 'err_rnn':
        records_fn = f'rnn_agent_records_{agent.model_type}-{trial_type}.pth'
        correct_rates_fn = f'correct_rates_{agent.model_type}-{trial_type}.pth'

    epochs_all = np.arange(n_epochs)
    losss = []
    correct_rates = []
    amrs = []

    episode_reps_rewarded = [torch.from_numpy(episode_reps_rewarded[idx]).type(
        DTYPE).to(device) for idx in range(len(episode_reps_rewarded))]
    records = {}
    for epoch in epochs_all:
        agent.train()
        losss_in_epoch = []
        act_matchs = []
        idx_all = list(range(len(episode_reps_rewarded)))
        random.shuffle(idx_all)
        pbar = tqdm(idx_all)

        h, o = agent.reset_h()
        workedtillact = []
        for idx in pbar:
            ep_rep_sample = episode_reps_rewarded[idx].data

            loss = 0
            act_match = 1
            trial_length = len(ep_rep_sample)
            actual_trial_length = 0
            actual_done = False
            for t in range(trial_length - 1):

                x = ep_rep_sample[t]
                y = ep_rep_sample[t + 1]
                if agent.esn:
                    h = h.data.detach()

                h, o = agent(h, x, o.data.detach())

                act_pred = torch.multinomial(
                    o[12:18], num_samples=1,
                    replacement=False).item()

                act_real = torch.argmax(y[12:18]).item()
                act_match *= float(act_pred == act_real)
                if act_match == 1 and actual_done is False:
                    actual_trial_length = t
                else:
                    actual_done = True
                loss_tmp = compute_rep_loss(o, y)
                loss += loss_tmp
                if agent.esn:
                    loss_tmp = loss_tmp + agent.w_penalty_weight * \
                        agent.compute_w_reg()
                    loss_tmp.backward()

            loss /= trial_length
            workedtillact.append(
                float(actual_trial_length >= dt2phase_ids[dt]['probe'][0])
            )
            if not agent.esn:
                loss = loss + agent.w_penalty_weight * agent.compute_w_reg()
                agent.optim.zero_grad()
                loss.backward()
                agent.optim.step()
            else:
                agent.optim.step()
                agent.optim.zero_grad()

            losss_in_epoch.append(loss.item())

            act_matchs.append(act_match)
            pbar.set_postfix({
                'model_type': agent.model_type,
                'loss': np.mean(losss_in_epoch),
                'act_match': np.mean(act_matchs),
                'workedtillact': f'{np.mean(workedtillact):.3f}',
            })

            h, o = agent.reset_h()

        losss.append(losss_in_epoch)
        episodes_eval, cr = gen_eval_episodes(
            agent, samples_required, trial_type, dt=dt, show_pbar=True)

        amr = np.mean(act_matchs)
        print(
            f'Epoch({epoch+1}), Trial type:{trial_type}. '
            f'loss: {np.mean(losss_in_epoch):.4f}, act_match={amr:.4f}, '
            f'correct rate(eval): {cr:.4f}')
        correct_rates.append(cr)
        amrs.append(amr)

        torch.save({
            'correct_rates': correct_rates,
            'losss': losss,
        }, agent.run_dir / correct_rates_fn)

        if eval_every_ep is True:
            episodes_eval, cr = gen_eval_episodes(
                agent, sample2test, trial_type, dt=dt,
                show_pbar=True, include_h=True, inclure_rew_info=True)

            print(f'correct rate(eval) for type {trial_type}: {cr:.3f}')
            records[trial_type] = episodes_eval
            records['dynamics'] = {
                'correct_rates': correct_rates,
                'losss': losss,
            }
            torch.save(records, agent.run_dir / records_fn)
            torch.save(records, agent.run_dir / f'{records_fn}.{epoch}')

        if cr >= cr_stop and epoch > 1:
            break

    if eval_every_ep is False:
        episodes_eval, cr = gen_eval_episodes(
            agent, sample2test, trial_type, dt=dt,
            show_pbar=True, include_h=True, inclure_rew_info=True)
        print(f'correct rate(eval) for type {trial_type}: {cr:.3f}')
        records[trial_type] = episodes_eval
        records['dynamics'] = {
            'correct_rates': correct_rates,
            'losss': losss,
        }
        torch.save(records, agent.run_dir / records_fn)

    torch.save(
        agent.state_dict(), agent.run_dir / 'agent.pth')

    return records_fn


@cli.command()
@click.option('--n_epochs_pre', default=20)
@click.option('--dt', default=250)
@click.option('--tau', default=5.0)
@click.option('--crs', default=0.9)
@click.option('--nh', default=256)
@click.option('--lr', default=5e-4)
@click.option('--force', default=False)
@click.option('--extra', default='', type=str)
@click.option('--load-pre', default=None, type=str)
@click.option('--timestamp', default=None, type=str)
@click.option('--model-type', default='all', type=str)
def full(n_epochs_pre, tau, dt, crs, nh, lr, force, extra, load_pre,
         timestamp, model_type):
    all_model_types = ('normal_rnn', 'err_rnn', 'esn')
    if model_type != 'all':
        assert model_type in all_model_types
        all_model_types = (model_type,)

    total_t_mature = 100000
    sim_dir = Path('data/sim')

    sim_dir.mkdir(exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print('timestamp:', timestamp)
    trial_types = ['prepotent', 'balanced']

    data_t = {}
    for trial_type in trial_types:
        data_fn = sim_dir / \
            f'axby_{trial_type}_{total_t_mature}_{extra}_{dt}.pth'

        if data_fn.exists() and not force:
            ep_r, cr = torch.load(data_fn)
        else:
            ep_r, cr = gen_training_samples(
                trial_type, total_t_mature, dt=dt)
            print(f'correct rate(trained) for {trial_type} trial:{cr:.4f}')
            torch.save((ep_r, cr), data_fn)
        data_t[trial_type] = (ep_r, cr)

    run_dir = sim_dir / f'{timestamp}-{extra}-{dt}'
    run_dir.mkdir(exist_ok=True)
    records_fns = []
    for mt in all_model_types:
        model = AgentNN(n_hid=nh, tau=tau, run_dir=run_dir,
                        device=device, model_type=mt)
        model.to(device)

        eval_every_eps = [False, False]
        n_epochs_all = [n_epochs_pre, 2]
        for trial_type, eval_every_ep, n_epochs in zip(
                trial_types, eval_every_eps, n_epochs_all):
            if load_pre is not None and trial_type == 'prepotent':
                print('loading pretrained model for prepotent trials')
                model.load_state_dict(torch.load(load_pre))
                continue

            optim = torch.optim.Adam(model.parameters(), lr=lr)
            model.optim = optim
            records_fn = train(model, n_epochs, data_t[trial_type][0],
                               data_t[trial_type][1],
                               dt, trial_type,
                               samples_required=300, cr_stop=crs,
                               eval_every_ep=eval_every_ep,
                               )
            records_fns.append(records_fn)



    print('timestamp:', timestamp)

    execute_analysis(
        run_dir,
        'data/exp_new_w_nids/DPX_statsOut.mat',
        'data/exp_new_w_nids/monkey_data',
        False,
        True,
        12,
    )


if __name__ == '__main__':
    cli()
