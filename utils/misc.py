import os, sys
import torch
import numpy as np
import h5py
import ntpath
import wandb

from utils.compat import ensure_d4rl_registered

def wandb_init(config_dict):
    wandb.init(
            config=config_dict,
            project=config_dict['wandb_project'],
            group=config_dict['wandb_group'],
            name=config_dict['wandb_name']
            )
    wandb.run.save()


def soft_update(target,source,tau):

    target_params_dict = dict(target.named_parameters())
    params_dict = dict(source.named_parameters())

    for key in target_params_dict:
        target_params_dict[key] = tau*params_dict[key] +\
                                    (1-tau)*target_params_dict[key]


    target.load_state_dict(target_params_dict)

def get_dataset(env):
    d4rl = ensure_d4rl_registered()

    path = os.path.expanduser('~/.d4rl/datasets')
    file_name = ntpath.basename(env.spec.kwargs['dataset_url'])
    filepath = os.path.join(path,file_name)
    if os.path.isfile(filepath) and ('pen' not in env.spec.name and 'maze' not in env.spec.name):

        hdf5_dataset = h5py.File(filepath,'r')
        dataset = {}

        for key in hdf5_dataset:
            if isinstance(hdf5_dataset[key],h5py.Dataset):
                dataset[key] = np.zeros(hdf5_dataset[key].shape)
                hdf5_dataset[key].read_direct(dataset[key])


    else:
        dataset = d4rl.qlearning_dataset(env)

   #dataset = d4rl.qlearning_dataset(env)

    return dataset

def batch_select_agents(tensor, agent_idx):
    tensor = torch.permute(tensor,(1,0,2))

    first_idx = torch.arange(agent_idx.shape[0])
    ##cant reassign to tensor whilst indexing!
    t = tensor[first_idx,agent_idx]

    return t


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Scalar(torch.nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = torch.nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant


def get_returns_to_go(agent):

    replay_buffer = agent.replay_buffer

    returns = []
    ep_ret, ep_len = 0, 0

    cur_rewards = []
    terminals = []
    N = len(replay_buffer)

    for t, (r,d) in enumerate(zip(replay_buffer.reward_memory,replay_buffer.terminal_memory)):
        ep_ret += float(r)
        cur_rewards.append(float(r))
        terminals.append(float(d))
        ep_len +=1

        is_last_step = (
                        (t== N-1) or
                         np.linalg.norm(
                             replay_buffer.state_memory[t + 1] - replay_buffer.next_state_memory[t]
                             )
                         > 1e-6
                         )
        if d or is_last_step:
            discounted_returns = [0] * ep_len
            prev_return = 0

            for i in reversed(range(ep_len)):
                discounted_returns[i] = cur_rewards[i] + agent.gamma*prev_return*(1-terminals[i])

                prev_return = discounted_returns[i]

            returns += discounted_returns
            ep_ret, ep_len = 0, 0
            cur_rewards = []
            terminal = []

    return torch.tensor(returns,dtype=torch.float,device=agent.device)
