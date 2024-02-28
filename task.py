import copy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from seq_pred_machine import episodes_post_process

TRIAL_TYPES = ['balanced', 'prepotent']


class AXBYEnv(gym.Env):
    def __init__(self, trial_set_type="balanced", dt=100):
        assert 500 % dt == 0
        self.dt = dt

        self.cue_probes = np.zeros((12, 12))
        np.fill_diagonal(self.cue_probes, 1.)
        self.cues = self.cue_probes[:6, :]
        self.probes = self.cue_probes[6:, :]
        self.empty_obs = np.zeros(12) + 1 / 12.


        self.comb = ['AX', 'AY', 'BX', 'BY']

        if trial_set_type == "balanced":
            self.probs = [0.25, 0.25, 0.25, 0.25]
        elif trial_set_type == "prepotent":
            self.probs = [0.69, 0.125, 0.125, 0.06]
        else:
            raise Exception('no such trial sets')

        self.action_space = spaces.Discrete(2 * 3)
        self._action2command = {
            0: [0, 0],
            1: [0, 1],
            2: [0, 2],
            3: [1, 0],
            4: [1, 1],
            5: [1, 2],
        }
        self._command2action = {
            tuple(v): k for k, v in self._action2command.items()}

    def reset(self):
        self.cue_prob = np.random.choice(self.comb, 1, p=self.probs)[0]
        if self.cue_prob[0] == 'A':
            self.cue = self.cues[0]

        elif self.cue_prob[0] == 'B':
            self.cue = self.cues[np.random.randint(1, 5)]
        else:
            raise ValueError
        if self.cue_prob[1] == 'X':
            self.probe = self.probes[0]

        elif self.cue_prob[1] == 'Y':
            self.probe = self.probes[np.random.randint(1, 5)]
        else:
            raise ValueError
        self.t = 0

        observation = self.empty_obs

        info = {'type': self.cue_prob}
        return observation, info

    def step(self, action):
        self.t += self.dt
        command = self._action2command[action]
        fixation = command[0]
        joystick = command[1]

        if 0 < self.t <= 500:
            observation = self.empty_obs
        elif 500 < self.t <= 1500:
            observation = self.cue
        elif 1500 < self.t <= 2500:
            observation = self.empty_obs
        elif 2500 < self.t <= 3000:
            observation = self.probe
        elif 3000 < self.t <= 4000:
            observation = self.empty_obs
        else:
            observation = self.empty_obs

        info = {'type': self.cue_prob}

        if fixation == 1:
            terminated = True
            reward = 0
            return observation, reward, terminated, False, info

        if self.t <= 2500:
            if joystick != 0:
                terminated = True
                reward = 0
            else:
                terminated = False
                reward = 0

        elif 2500 < self.t <= 4000:
            if self.cue_prob == "AX" and joystick == 1:
                terminated = True
                reward = 1
            elif self.cue_prob != "AX" and joystick == 2:
                terminated = True
                reward = 1
            else:
                terminated = False
                reward = 0
        else:
            terminated = True
            reward = 0

        return observation, reward, terminated, False, info


class MatureAgent:
    def __init__(self, env, p_f, p_j):
        self.reset()
        self.p_f = p_f
        self.p_j = p_j

        self._env = env
        pass

    def select_act(self, obs):
        fixation = np.random.choice([0, 1], p=[self.p_f, 1 - self.p_f])

        joystick = np.random.choice(
            [0, 1, 2], p=[self.p_j, (1 - self.p_j) / 2, (1 - self.p_j) / 2])
        if 2600 < self._env.t <= 4000:
            if self._env.cue_prob[0] == "B":
                pj = max(0.999, self.p_j)
                joystick = np.random.choice(
                    [0, 1, 2], p=[(1 - pj) / 2, (1 - pj) / 2, pj])

            elif self._env.cue_prob == "AX":
                joystick = np.random.choice(
                    [0, 1, 2], p=[(1 - self.p_j) / 2, self.p_j, (1 - self.p_j) / 2])
            elif self._env.cue_prob == "AY":
                pj = 0.33
                joystick = np.random.choice(
                    [0, 1, 2], p=[(1 - pj) / 2, (1 - pj) / 2, pj])
            else:
                raise Exception
        self.internal_t += 1

        return self._env._command2action[(fixation, joystick)]

    def reset(self):
        self.internal_t = 0


def gen_training_episodes(total_t, trial_type, p_f=0.999, p_j=0.995, dt=250):
    assert trial_type in TRIAL_TYPES
    env = AXBYEnv(trial_type, dt=dt)
    agent = MatureAgent(env, p_f, p_j)

    episodes = []

    obs, info = env.reset()
    act = 0
    reward = 0
    done = False
    episode = [(obs.copy(), act, reward, done, info)]

    for t in range(total_t):
        act = agent.select_act(obs)
        obs_next, reward, terminated, truncated, info = env.step(act)
        done = truncated or terminated

        episode.append(
            (obs.copy(), act, reward, done, info)
        )
        obs = obs_next
        if done:
            episodes.append(copy.deepcopy(episode))

            obs, info = env.reset()
            act = 0
            reward = 0
            done = False
            episode = [(obs.copy(), act, reward, done, info)]

    episode_reps = episodes_post_process(episodes)
    return episode_reps
