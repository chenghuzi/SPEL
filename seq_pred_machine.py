import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from collections import defaultdict
import json
DTYPE = torch.FloatTensor
device = 'cuda' if torch.cuda.is_available() else "cpu"


class AgentNN(nn.Module):
    def __init__(self, n_hid, tau, run_dir: Path,
                 optim=None, device='cpu', model_type='err_rnn'):
        super().__init__()
        self.device = device
        self.n_hid = n_hid
        self.n_out = 22
        self.n_in = 22
        self.model_type = model_type

        with open('config.json') as f:
            m_config = json.load(f)[model_type]

        if self.model_type == 'err_rnn':
            self.esn = False
            self.err_as_inputs = True
        elif self.model_type == 'normal_rnn':
            self.esn = False
            self.err_as_inputs = False
        elif self.model_type == 'esn':
            self.esn = False
            self.err_as_inputs = False
        else:
            raise ValueError('No such model type')

        self.i2h = nn.Sequential(
            nn.Linear(self.n_in, self.n_hid),
        )

        self.act_f = nn.Tanh()

        self.h2o = nn.Sequential(
            nn.Linear(self.n_hid, self.n_out),
        )

        self.h2h = nn.Linear(self.n_hid, self.n_hid)
        if self.esn:
            self.h2h.weight.requires_grad = False
            self.h2h.bias.requires_grad = False

        self.tau = tau
        self.i2o = nn.Sequential(
            nn.Linear(self.n_in, self.n_out),
        )

        self.internal_noise_level = m_config["internal_noise_level"]
        self.external_noise_level = m_config["external_noise_level"]

        self.optim = optim
        self.w_penalty_weight = m_config["w_penalty_weight"]
        self.h_penalty_weight = m_config["h_penalty_weight"]
        self.run_dir = run_dir

    def decode(self, h, x=None):
        if x is None:
            out = self.h2o(h)
        else:
            out = self.h2o(h)

        out[12:18] = F.softmax(out[12:18], dim=-1)
        out[18:20] = F.softmax(out[18:20], dim=-1)
        out[20:22] = F.softmax(out[20:22], dim=-1)
        return out

    def forward(self, h, x, o_prev):
        """
        h: hidden state
        x: input
        o_prev: previous observation
        Notice that the external_noise_level is very important.
        Be careful when you change it.
        """
        internal_noise = self.internal_noise_level * torch.normal(
            torch.zeros_like(h), torch.ones_like(h))
        dnew = 1. / self.tau

        x = x + self.external_noise_level * torch.normal(
            torch.zeros_like(x), torch.ones_like(x)).to(self.device)
        if self.err_as_inputs:
            x = x - o_prev

        if self.esn:
            with torch.no_grad():
                h_update = self.h2h(h)
        else:
            h_update = self.h2h(h)
        inputs_all = self.act_f(h_update + self.i2h(x))
        h_new = (1 - dnew) * h + dnew * inputs_all + internal_noise

        out = self.decode(h_new, x)
        return h_new, out

    def select_action(self, h, x, o_prev):
        h_new, out = self.forward(h, x, o_prev)
        act_probs = out[12:18]
        act = torch.multinomial(
            act_probs, num_samples=1, replacement=False).data.item()

        return act, h_new, out

    def reset_h(self):
        init_h = 2 * (torch.rand(self.n_hid).to(self.device) - 0.5)
        init_o = self.decode(init_h)
        return init_h, init_o

    def compute_w_reg(self):
        w_reg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'weight' in name:
                w_reg = w_reg + \
                    torch.linalg.norm(param, ord=2) / param.numel()
        return w_reg


def encode_inputs(obs, act, reward, done, to_torch=True, dtype=DTYPE):
    """Encode raw sensory inputs, act, rewards, etc. for the model.
    """
    x = np.zeros(22)
    x[0:12] = obs
    x[12 + act] = 1.
    x[18 + reward] = 1.
    x[20 + done] = 1.
    if to_torch:
        return torch.from_numpy(x).type(dtype).to(device)
    else:
        return x


def compute_rep_loss(od, yd):
    s_loss = F.mse_loss(od[0:12], yd[0:12])
    act_loss = F.mse_loss(od[12:18], yd[12:18])
    rew_loss = F.mse_loss(od[18:20], yd[18:20])
    done_loss = F.mse_loss(od[20:22], yd[20:22])
    loss = s_loss + act_loss + rew_loss + done_loss
    return loss


def episodes_post_process(episodes):
    dim = 12 + 6 + 2 + 2
    ep_reps = []
    ep_reps_statistics = defaultdict(list)
    for ep in episodes:
        ep_len = len(ep)
        ep_rep = np.zeros((ep_len, dim))
        act_prev = 3
        acts = []
        for t in range(ep_len):
            obs, act, reward, done, info = ep[t]

            ep_rep[t] = encode_inputs(
                obs, act, reward, done, to_torch=False)
            if act_prev != act:
                act_prev = act
            acts.append(act)

        ep_reps.append((
            ep_rep.copy(),
            reward
        ))
        ep_reps_statistics[info['type']].append(ep_len)

    ep_reps_statistics_mean = {}
    for k, v in ep_reps_statistics.items():
        ep_reps_statistics_mean[k] = {
            'mean': np.mean(v),
            'std': np.std(v),
        }
    print(
        f'AX: {ep_reps_statistics_mean["AX"]}\n'
        f'AY: {ep_reps_statistics_mean["AY"]}\n'
        f'BX: {ep_reps_statistics_mean["BX"]}\n'
        f'BY: {ep_reps_statistics_mean["BY"]}\n'
    )

    return ep_reps
