# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:20:39 2018
@author: fy4bc
"""
import time
import numpy as np
import torch
from dgl.graph import DGLGraph
from log_utils import mean_val, logger
from k_cut import *
from dataclasses import dataclass
import itertools
from toy_models.Qiter import vis_g, state2QtableKey, QtableKey2state


def to_cuda(G_, copy=True):
    if copy:
        G = dc(G_)
    else:
        G = G_
    if 'adj' in G.ndata.keys():
        G.ndata['adj'] = G.ndata['adj'].cuda()
    G.ndata['label'] = G.ndata['label'].cuda()
    G.edata['d'] = G.edata['d'].cuda()
    G.edata['e_type'] = G.edata['e_type'].cuda()
    return G


class EpisodeHistory:
    def __init__(self, g, max_episode_len, action_type='swap'):
        self.action_type = action_type
        self.init_state = dc(g)

        self.n = g.number_of_nodes()
        self.max_episode_len = max_episode_len
        self.episode_len = 0
        self.action_seq = []
        self.action_indices = []
        self.reward_seq = []
        self.q_pred = []
        self.action_candidates = []
        self.enc_state_seq = []
        self.sub_reward_seq = []
        if self.action_type == 'swap':
            self.label_perm = torch.tensor(range(self.n)).unsqueeze(0)
            self.enc_state_seq.append(state2QtableKey(self.init_state.ndata['label'].argmax(dim=1).cpu().numpy()))
        if self.action_type == 'flip':
            self.label_perm = self.init_state.ndata['label'].nonzero()[:, 1].unsqueeze(0)
        self.best_gain_sofar = 0
        self.current_gain = 0
        self.loop_start_position = 0

    def perm_label(self, label, action):
        label = dc(label)
        if self.action_type == 'swap':
            tmp = dc(label[action[0]])
            label[action[0]] = label[action[1]]
            label[action[1]] = tmp
        if self.action_type == 'flip':
            label[action[0]] = action[1]
        return label.unsqueeze(0)

    def write(self, action, action_idx, reward, q_val=None, actions=None, state_enc=None, sub_reward=None, loop_start_position=None):

        new_label = self.perm_label(self.label_perm[-1, :], action)

        self.action_seq.append(action)

        self.action_indices.append(action_idx)

        self.reward_seq.append(reward)

        self.q_pred.append(q_val)

        self.action_candidates.append(actions)

        self.enc_state_seq.append(state_enc)

        self.sub_reward_seq.append(sub_reward)

        self.loop_start_position = loop_start_position

        self.label_perm = torch.cat([self.label_perm, new_label], dim=0)

        self.episode_len += 1

    def wrap(self):
        self.reward_seq = torch.tensor(self.reward_seq)
        self.empl_reward_seq = torch.tensor(self.empl_reward_seq)
        self.label_perm = self.label_perm.long()


@dataclass
class sars:
    # TODO: save space for s1
    s0: DGLGraph
    a: tuple
    r: float
    s1: DGLGraph
    rollout_r: torch.tensor


class DQN:
    def __init__(self, problem
                 , action_type='swap'
                 , gamma=1.0, eps=0.1, lr=1e-4, action_dropout=1.0
                 , sample_batch_episode=False
                 , replay_buffer_max_size=5000
                 , epi_len=50, new_epi_batch_size=10
                 , edge_info='adj_weight'
                 , readout='mlp'
                 , explore_method='epsilon_greedy'
                 , priority_sampling='False'
                 , clip_target=False):

        self.problem = problem
        self.action_type = action_type
        self.G = problem.g  # the graph
        self.k = problem.k  # num of clusters
        self.ajr = problem.adjacent_reserve  # degree of node in graph
        self.hidden_dim = problem.hidden_dim  # hidden dimension for node representation
        self.n = problem.N  # num of nodes
        self.eps = eps  # constant for exploration in dqn
        self.edge_info = edge_info
        self.explore_method = explore_method
        self.clip_target = clip_target
        self.model = DQNet(k=self.k, n=self.n, num_head=2, hidden_dim=self.hidden_dim, edge_info=self.edge_info, readout=readout).cuda()
        self.model_target = dc(self.model)
        self.gamma = gamma  # reward decay const
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.sample_batch_episode = sample_batch_episode
        self.experience_replay_buffer = []
        self.replay_buffer_max_size = replay_buffer_max_size
        self.buf_epi_len = epi_len  # 50
        self.new_epi_batch_size = new_epi_batch_size  # 10
        self.cascade_replay_buffer = [[] for _ in range(self.buf_epi_len)]
        self.cascade_replay_buffer_weight = torch.zeros((self.buf_epi_len, self.new_epi_batch_size))
        self.stage_max_sizes = [self.replay_buffer_max_size // self.buf_epi_len] * self.buf_epi_len  # [100, 100, ..., 100]
        # self.stage_max_sizes = list(range(100,100+4*50, 4))
        self.buffer_actual_size = sum(self.stage_max_sizes)
        self.priority_sampling = priority_sampling
        self.cascade_buffer_kcut_value = torch.zeros((self.buf_epi_len, self.new_epi_batch_size))
        self.action_dropout = action_dropout
        self.log = logger()
        self.Q_err = 0  # Q error
        self.log.add_log('tot_return')
        self.log.add_log('Q_error')
        self.log.add_log('entropy')
        self.log.add_log('R_signal')

    def _updata_lr(self, step, max_lr, min_lr, decay_step):
        for g in self.optimizer.param_groups:
            g['lr'] = max(max_lr / ((max_lr / min_lr) ** (step / decay_step) ), min_lr)

    def run_batch_episode(self, target_bg=None, action_type='swap', gnn_step=3, episode_len=50, batch_size=10, rollout_step=1):

        sum_r = 0

        if target_bg is None:
            bg = to_cuda(self.problem.gen_batch_graph(batch_size=batch_size))
        else:
            bg = to_cuda(self.problem.gen_target_batch_graph(target_bg=target_bg, batch_size=batch_size))

        num_actions = self.problem.get_legal_actions(action_type=action_type, action_dropout=self.action_dropout).shape[0]
        action_mask = torch.tensor(range(0, num_actions * batch_size, num_actions)).cuda()

        explore_dice = (torch.rand(episode_len, batch_size) < self.eps)
        explore_replace_mask = explore_dice.nonzero()  #
        explore_step_offset = torch.cat([torch.zeros([1], dtype=torch.long), torch.cumsum(explore_dice.sum(dim=1), dim=0)], dim=0)
        explore_replace_actions = torch.randint(high=num_actions, size=(explore_replace_mask.shape[0], )).cuda()

        t = 0
        while t < episode_len:

            batch_legal_actions = self.problem.get_legal_actions(state=bg, action_type=action_type, action_dropout=self.action_dropout).cuda()

            # epsilon greedy strategy
            # TODO: cannot dc bg once being forwarded.
            # TODO: multi - gpu parallelization
            _, _, _, Q_sa = self.model(dc(bg), batch_legal_actions, action_type=action_type, gnn_step=gnn_step)

            best_actions = Q_sa.view(-1, num_actions).argmax(dim=1)

            explore_episode_indices = explore_replace_mask[explore_step_offset[t]: explore_step_offset[t + 1]][:, 1]
            explore_actions = explore_replace_actions[explore_step_offset[t]: explore_step_offset[t + 1]]
            best_actions[explore_episode_indices] = explore_actions
            best_actions += action_mask

            actions = batch_legal_actions[best_actions]

            # update bg inplace and calculate batch rewards
            g0 = [g for g in dgl.unbatch(dc(bg))]  # current_state
            _, rewards = self.problem.step_batch(states=bg, action=actions, action_type=action_type)
            g1 = [g for g in dgl.unbatch(dc(bg))]  # after_state

            if self.sample_batch_episode:
                self.experience_replay_buffer.extend([sars(g0[i], actions[i], rewards[i], g1[i], torch.zeros((rollout_step)).cuda()) for i in range(batch_size)])
            else:  # using cascade buffer

                self.cascade_replay_buffer[t].extend([sars(g0[i], actions[i], rewards[i], g1[i], torch.zeros((rollout_step)).cuda()) for i in range(batch_size)])

                if self.priority_sampling:
                    # compute prioritized weights
                    batch_legal_actions = self.problem.get_legal_actions(state=bg, action_type=action_type, action_dropout=self.action_dropout).cuda()
                    _, _, _, Q_sa_next = self.model(dc(bg), batch_legal_actions, action_type=action_type, gnn_step=gnn_step)
                    delta = Q_sa[best_actions] - (rewards + self.gamma * Q_sa_next.view(-1, num_actions).max(dim=1).values)
                    # delta = (Q_sa[best_actions] - (rewards + self.gamma * Q_sa_next.view(-1, num_actions).max(dim=1).values)) / (torch.clamp(torch.abs(Q_sa[best_actions]),0.1))
                    self.cascade_replay_buffer_weight[t, :batch_size] = torch.abs(delta.detach())
            R = [reward.item() for reward in rewards]
            sum_r += sum(R)

            t += 1

        self.log.add_item('tot_return', sum_r)

        return R

    def sample_actions_from_q(self, Q_sa, num_actions, batch_size, Temperature=1.0, eps=None, top_k=1):

        if self.explore_method == 'epsilon_greedy':
            # len = batch_size * topk  (g0_top1, g1_top1, ..., g0_top2, ...)
            best_actions = Q_sa.view(batch_size, num_actions).topk(k=top_k, dim=1).indices.t().flatten()

        if self.explore_method == 'softmax' or self.explore_method == 'soft_dqn':

            best_actions = torch.multinomial(F.softmax(Q_sa.view(-1, num_actions) / Temperature), 1).view(-1)

        if eps is None:
            eps = self.eps
        explore_replace_mask = (torch.rand(batch_size * top_k) < eps).nonzero()
        explore_actions = torch.randint(high=num_actions, size=(explore_replace_mask.shape[0], )).cuda()
        best_actions[explore_replace_mask[:, 0]] = explore_actions
        # add action batch offset
        best_actions += torch.tensor(range(0, num_actions * batch_size, num_actions)).repeat(top_k).cuda()
        return best_actions

    def rollout(self, bg, num_actions, rollout_step, top_num=5):

        batch_size = self.new_epi_batch_size * self.buf_epi_len * top_num
        rollout_rewards = torch.zeros((batch_size, rollout_step)).cuda()
        for step in range(rollout_step):
            batch_legal_actions = self.problem.get_legal_actions(state=bg, action_type=self.action_type,
                                                                 action_dropout=self.action_dropout).cuda()

            _, _, _, Q_sa = self.model(bg, batch_legal_actions, action_type=self.action_type)

            chosen_actions = self.sample_actions_from_q(Q_sa, num_actions, batch_size, eps=0.0, top_k=1)
            _actions = batch_legal_actions[chosen_actions]

            # update bg inplace and calculate batch rewards
            _, _rewards = self.problem.step_batch(states=bg, action_type=self.action_type, action=_actions)

            rollout_rewards[:, step] = _rewards
            # print('step', step, rewards)
        return rollout_rewards

    def run_cascade_episode(self, target_bg=None, action_type='swap', gnn_step=3, q_step=1, rollout_step=0, verbose=False):

        sum_r = 0

        T0 = time.time()

        # generate new start states
        if target_bg is None:
            new_graphs = dgl.unbatch(to_cuda(self.problem.gen_batch_graph(batch_size=self.new_epi_batch_size), copy=False))
        else:
            new_graphs = dgl.unbatch(to_cuda(self.problem.gen_target_batch_graph(target_bg=target_bg, batch_size=self.new_epi_batch_size), copy=False))
        if verbose:
            T1 = time.time(); print('t1', T1 - T0)

        # extend previous states
        new_graphs.extend(list(itertools.chain(*[[tpl.s1 for tpl in self.cascade_replay_buffer[i][-self.new_epi_batch_size:]] for i in range(self.buf_epi_len-1)])))
        if verbose:
            T2 = time.time(); print('t2', T2 - T1)

        # make batch and copy new states
        bg = to_cuda(dgl.batch(new_graphs))
        if verbose:
            T3 = time.time(); print('t3', T3 - T2)

        batch_size = self.new_epi_batch_size * self.buf_epi_len
        num_actions = self.problem.get_legal_actions(action_type=action_type, action_dropout=self.action_dropout).shape[0]
        if isinstance(self.problem.g, BatchedDGLGraph):
            num_actions //= self.problem.g.batch_size
        if verbose:
            T4 = time.time(); print('t4', T4 - T3)
        # recent_states = list(itertools.chain(
        #     *[[tpl.recent_states for tpl in self.cascade_replay_buffer[t][-self.new_epi_batch_size:]] for t in
        #       range(self.buf_epi_len - 1)]))
        #
        # batch_legal_actions = self.prune_actions(bg, recent_states)


        batch_legal_actions = self.problem.get_legal_actions(state=bg, action_type=self.action_type,
                                                                 action_dropout=self.action_dropout).cuda()
        if verbose:
            T5 = time.time(); print('t5', T5 - T4)
        # epsilon greedy strategy
        # TODO: multi-gpu parallelization
        _, _, _, Q_sa = self.model(bg, batch_legal_actions, action_type=action_type, gnn_step=gnn_step)
        if verbose:
            T6 = time.time(); print('t6', T6 - T5)

        if not rollout_step:
            # TODO: can alter explore strength according to kcut_valueS
            chosen_actions = self.sample_actions_from_q(Q_sa, num_actions, batch_size, Temperature=self.eps)
            actions = batch_legal_actions[chosen_actions]
            rollout_rewards = torch.zeros(batch_size, 1)
        else:
            top_num = 10  # rollout for how many top actions
            rollout_bg = dgl.batch([bg] * top_num)

            # chosen_actions - len = batch_size * topk
            chosen_actions = self.sample_actions_from_q(Q_sa, num_actions, batch_size, Temperature=self.eps, top_k=top_num)

            topk_actions = batch_legal_actions[chosen_actions]

            bg1, rewards1 = self.problem.step_batch(states=rollout_bg, action_type=action_type, action=topk_actions)

            rollout_rewards = self.rollout(bg=bg1, num_actions=num_actions, rollout_step=rollout_step, top_num=top_num)

            # select actions based on rollout rewards
            # rollout_selected_actions = torch.cat([rewards1.view(-1, 1), rollout_rewards], dim=1)\
            rollout_selected_actions = torch.cat([rollout_rewards], dim=1)\
                .cumsum(dim=1).max(dim=1)\
                .values.view(top_num, -1)\
                .argmax(dim=0) * batch_size + torch.tensor(range(batch_size)).cuda()

            # update bg inplace and calculate batch rewards
            actions = topk_actions[rollout_selected_actions, :]
            # rewards = rewards1[rollout_selected_actions]
            rollout_rewards = rollout_rewards[rollout_selected_actions, :]

        # update bg inplace and calculate batch rewards
        _, rewards = self.problem.step_batch(states=bg, action_type=action_type, action=actions)

        g0 = new_graphs  # current_state
        g1 = [g for g in dgl.unbatch(bg)]  # after_state

        [self.cascade_replay_buffer[t].extend(
            [sars(g0[j+t*self.new_epi_batch_size]
            , actions[j+t*self.new_epi_batch_size]
            , rewards[j+t*self.new_epi_batch_size]
            , g1[j+t*self.new_epi_batch_size]
            , rollout_rewards[j+t*self.new_epi_batch_size, :])
            for j in range(self.new_epi_batch_size)])
         for t in range(self.buf_epi_len)]

        if self.priority_sampling:
            # compute prioritized weights
            batch_legal_actions = self.problem.get_legal_actions(state=bg, action_type=action_type, action_dropout=self.action_dropout).cuda()
            _, _, _, Q_sa_next = self.model(dc(bg), batch_legal_actions, action_type=action_type, gnn_step=gnn_step)

            delta = Q_sa[chosen_actions] - (rewards + self.gamma * Q_sa_next.view(-1, num_actions).max(dim=1).values)
            # delta = (Q_sa[chosen_actions] - (rewards + self.gamma * Q_sa_next.view(-1, num_actions).max(dim=1).values)) / torch.clamp(torch.abs(Q_sa[chosen_actions]),0.1)
            self.cascade_replay_buffer_weight = torch.cat([self.cascade_replay_buffer_weight, torch.abs(delta.detach().cpu().view(self.buf_epi_len, self.new_epi_batch_size))], dim=1).detach()
            # print(self.cascade_replay_buffer_weight)

        R = [reward.item() for reward in rewards]
        sum_r += sum(R)

        self.log.add_item('tot_return', sum_r)


        return R

    def sample_from_buffer(self, batch_size, q_step, gnn_step):

        batch_size = min(batch_size, len(self.experience_replay_buffer))

        sample_buffer = np.random.choice(self.experience_replay_buffer, batch_size, replace=False)
        # make batches
        batch_begin_state = dgl.batch([tpl.s0 for tpl in sample_buffer])
        batch_end_state = dgl.batch([tpl.s1 for tpl in sample_buffer])
        R = [tpl.r.unsqueeze(0) for tpl in sample_buffer]
        batch_begin_action = torch.cat([tpl.a.unsqueeze(0) for tpl in sample_buffer], axis=0)
        batch_end_action = self.problem.get_legal_actions(state=batch_end_state, action_type=self.action_type, action_dropout=self.action_dropout).cuda()
        action_num = batch_end_action.shape[0] // batch_begin_action.shape[0]

        # only compute limited number for Q_s1a
        # TODO: multi-gpu parallelization
        _, _, _, Q_s1a_ = self.model(batch_begin_state, batch_begin_action, action_type=self.action_type, gnn_step=gnn_step)
        _, _, _, Q_s2a = self.model_target(batch_end_state, batch_end_action, action_type=self.action_type, gnn_step=gnn_step)


        q = self.gamma ** q_step * Q_s2a.view(-1, action_num).max(dim=1).values - Q_s1a_
        Q = q.unsqueeze(0)

        return torch.cat(R), Q

    def sample_from_cascade_buffer(self, batch_size, q_step, gnn_step, rollout_step=0, verbose=True):

        batch_size = min(batch_size, len(self.cascade_replay_buffer[0]) * self.buf_epi_len)

        T0 = time.time()
        if self.priority_sampling:
            nc = self.cascade_replay_buffer_weight.shape[1]
            selected_indices = torch.multinomial((0.1 + self.cascade_replay_buffer_weight.flatten()) ** 1.0, batch_size)
            sample_buffer = [self.cascade_replay_buffer[indices // nc][indices % nc] for indices in selected_indices]
        else:
            if False:
                # importance sampling according to S
                nc = self.cascade_buffer_kcut_value.shape[1]
                print('sample weight:', torch.sum(1.0 / self.cascade_buffer_kcut_value ** 2, dim=1))
                selected_indices = torch.multinomial((1.0 / self.cascade_buffer_kcut_value.flatten() ** 2), batch_size)
                sample_buffer = [self.cascade_replay_buffer[indices // nc][indices % nc] for indices in
                                 selected_indices]
            else:
                batch_sizes = [
                    min(batch_size * self.stage_max_sizes[i] // self.buffer_actual_size, len(self.cascade_replay_buffer[0]))
                    for i in range(self.buf_epi_len)]
                sample_buffer = list(itertools.chain(*[np.random.choice(a=self.cascade_replay_buffer[i]
                                                                    , size=batch_sizes[i]
                                                                    , replace=False
                                                                    # , p=
                                                                    ) for i in range(self.buf_epi_len)]))
        if verbose:
            T1 = time.time(); print('sampling:', T1-T0)

        # make batches
        batch_begin_state = dgl.batch([tpl.s0 for tpl in sample_buffer])
        T11 = time.time();
        print('mk batch:', T11 - T1)
        batch_end_state = dgl.batch([tpl.s1 for tpl in sample_buffer])
        T111 = time.time();
        print('mk batch:', T111 - T11)

        # _, rewards = self.problem.step_batch(states=bg, action=actions, action_type=action_type)
        # g1 = [g for g in dgl.unbatch(dc(bg))]  # after_state

        R = [tpl.r.unsqueeze(0) for tpl in sample_buffer]

        if rollout_step:
            rollout_R = torch.cat([tpl.rollout_r.unsqueeze(0) for tpl in sample_buffer])
            rollout_episode_R = rollout_R[:, 1:].cumsum(dim=1).max(dim=1).values

        batch_begin_action = torch.cat([tpl.a.unsqueeze(0) for tpl in sample_buffer], axis=0)
        batch_end_action = self.problem.get_legal_actions(state=batch_end_state, action_type=self.action_type, action_dropout=self.action_dropout).cuda()
        action_num = batch_end_action.shape[0] // batch_begin_action.shape[0]
        if verbose:
            T2 = time.time(); print('make batch:', T2 - T1)

        # only compute limited number for Q_s1a
        # TODO: multi-gpu parallelization
        _, _, _, Q_s1a_ = self.model(batch_begin_state, batch_begin_action, action_type=self.action_type, gnn_step=gnn_step)
        if verbose:
            T3 = time.time(); print('forward:', T3 - T2)
        _, _, _, Q_s2a = self.model_target(batch_end_state, batch_end_action, action_type=self.action_type, gnn_step=gnn_step)
        if verbose:
            T4 = time.time(); print('forward target:', T4 - T3)

        chosen_actions = self.sample_actions_from_q(Q_s2a, action_num, batch_size, Temperature=self.eps)

        if False and rollout_step:
            q = torch.max(self.gamma ** q_step * Q_s2a[chosen_actions], rollout_episode_R) - Q_s1a_
        else:
            q = self.gamma ** q_step * Q_s2a[chosen_actions] - Q_s1a_

        if verbose:
            T5 = time.time(); print('choose action from q:', T5 - T4)

        if self.priority_sampling:
            # print(self.cascade_replay_buffer_weight.view(-1)[selected_indices])
            self.cascade_replay_buffer_weight.flatten()[selected_indices] = torch.abs(q).cpu()
            # self.cascade_replay_buffer_weight.flatten()[selected_indices] = torch.abs(q / torch.clamp(torch.abs(Q_s1a_),0.1)).cpu()
            print(self.cascade_replay_buffer_weight.flatten()[selected_indices])

        Q = q.unsqueeze(0)

        return torch.cat(R), Q

    def back_loss(self, R, Q, update_model=True):


        L = torch.pow(R.cuda() + Q, 2).sum()
        L.backward(retain_graph=True)

        self.Q_err += L.item()

        if update_model:
            self.optimizer.step()

            self.optimizer.zero_grad()
            self.log.add_item('Q_error', self.Q_err)
            self.Q_err = 0
            self.log.add_item('entropy', 0)

    def train_dqn(self, target_bg=None, epoch=0, batch_size=16, num_episodes=10, episode_len=50, gnn_step=10, q_step=1, rollout_step=0, ddqn=False):
        """
        :param batch_size:
        :param num_episodes:
        :param episode_len: #steps in each episode
        :param gnn_step: #iters when running gnn
        :param q_step: reward delay step
        :param ddqn: train in ddqn mode
        :return:
        """
        if self.sample_batch_episode:
            T3 = time.time()
            self.run_batch_episode(action_type=self.action_type, gnn_step=gnn_step, episode_len=episode_len,
                                   batch_size=num_episodes)
            T4 = time.time()

            # trim experience replay buffer
            self.trim_replay_buffer(epoch)

            R, Q = self.sample_from_buffer(batch_size=batch_size, q_step=q_step, gnn_step=gnn_step)

            T6 = time.time()
        else:
            T3 = time.time()
            if epoch == 0:
                self.run_batch_episode(target_bg=target_bg, action_type=self.action_type, gnn_step=gnn_step, episode_len=self.buf_epi_len,
                                   batch_size=self.new_epi_batch_size, rollout_step=rollout_step)
                # change step batch mask helper
                self.problem.sample_episode = self.buf_epi_len * self.new_epi_batch_size
                self.problem.gen_step_batch_mask()
            else:
                self.run_cascade_episode(target_bg=target_bg, action_type=self.action_type, gnn_step=gnn_step, q_step=q_step, rollout_step=rollout_step)
            T4 = time.time()
            # trim experience replay buffer
            self.trim_replay_buffer(epoch)

            R, Q = self.sample_from_cascade_buffer(batch_size=batch_size, q_step=q_step, rollout_step=rollout_step, gnn_step=gnn_step)

            T6 = time.time()
        # 0.5s
        self.back_loss(R, Q, update_model=True)
        T7 = time.time()
        del R, Q
        torch.cuda.empty_cache()

        self._updata_lr(step=epoch, max_lr=2e-3, min_lr=1e-3, decay_step=10000)

        print('Rollout time:', T4-T3)
        print('Sample and forward time', T6-T4)
        print('Backloss time', T7-T6)

        return self.log

    def trim_replay_buffer(self, epoch):
        if len(self.experience_replay_buffer) > self.replay_buffer_max_size:
            self.experience_replay_buffer = self.experience_replay_buffer[-self.replay_buffer_max_size:]

        if epoch * self.buf_epi_len * self.new_epi_batch_size > self.replay_buffer_max_size:
            for i in range(self.buf_epi_len):
                self.cascade_replay_buffer[i] = self.cascade_replay_buffer[i][-self.stage_max_sizes[i]:]
        # self.cascade_replay_buffer_weight = self.cascade_replay_buffer_weight[:, -self.stage_max_sizes[0]:]
        # self.cascade_buffer_kcut_value = self.cascade_buffer_kcut_value[:, -self.stage_max_sizes[0]:]


    def update_target_net(self):
        self.model_target.load_state_dict(self.model.state_dict())
