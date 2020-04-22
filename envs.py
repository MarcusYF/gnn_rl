from graph_handler import *


def peek_greedy_reward(states, actions=None, action_type='swap'):
    """
    :param states: LightGraph
    :param actions:
    :param action_type:
    :return:
    """
    batch_size = states.batch_size
    n = states.n
    bn = batch_size * n

    if actions is None:
        actions = get_legal_actions(states=states, action_type=action_type, action_dropout=1.0, pause_action=True)

    group_matrix = torch.bmm(states.ndata['label'].view(batch_size, n, -1),
                             states.ndata['label'].view(batch_size, n, -1).transpose(1, 2)).view(bn, n)
    num_action = actions.shape[0] // batch_size

    action_mask = torch.tensor(range(0, bn, n)).unsqueeze(1).expand(batch_size, 2).repeat(1, num_action).view(
        num_action * batch_size, -1)
    if states.in_cuda:
        actions_ = actions + action_mask.cuda()
    else:
        actions_ = actions + action_mask
    #  (b, n, n)
    #  (b * num_action, n)
    rewards = (states.ndata['adj'][actions_[:, 0], :] * (
                group_matrix[actions_[:, 0], :] - group_matrix[actions_[:, 1], :])).sum(dim=1) \
              + (states.ndata['adj'][actions_[:, 1], :] * (
                group_matrix[actions_[:, 1], :] - group_matrix[actions_[:, 0], :])).sum(dim=1) \
              + 2 * states.ndata['adj'][actions_[:, 0], actions[:, 1]]

    return rewards


def get_legal_actions(states, action_type='swap', action_dropout=1.0, pause_action=True):
    """
    :param states: LightGraph
    :param action_type:
    :param action_dropout:
    :param pause_action:
    :return:
    """
    if action_type == 'flip':

        legal_actions = torch.nonzero(1 - states.ndata['label'])
        num_actions = legal_actions.shape[0] // states.batch_size

        mask = torch.tensor(range(0, states.n * states.batch_size, states.n)).repeat(num_actions).view(-1, states.batch_size).t().flatten()
        if states.in_cuda:
            legal_actions[:, 0] -= mask.cuda()
        else:
            legal_actions[:, 0] -= mask

        if action_dropout < 1.0:
            maintain_actions = int(num_actions * action_dropout)
            maintain = [np.random.choice(range(_ * num_actions, (_ + 1) * num_actions), maintain_actions, replace=False) for _ in range(states.batch_size)]
            legal_actions = legal_actions[torch.tensor(maintain).flatten(), :]
        if pause_action:
            legal_actions = legal_actions.reshape(states.batch_size, -1, 2)
            legal_actions = torch.cat([legal_actions, (legal_actions[:, 0] * 0 ).unsqueeze(1)], dim=1).view(-1, 2)

    if action_type == 'swap':

        n = states.n
        mask = torch.bmm(states.ndata['label'].view(states.batch_size, n, -1),
                         states.ndata['label'].view(states.batch_size, n, -1).transpose(1, 2))
        legal_actions = torch.triu(1 - mask).nonzero()[:, 1:3]  # tensor (270, 2)

        if action_dropout < 1.0:
            num_actions = legal_actions.shape[0] // states.batch_size
            maintain_actions = int(num_actions * action_dropout)
            maintain = [np.random.choice(range(_ * num_actions, (_ + 1) * num_actions), maintain_actions, replace=False) for _ in range(states.batch_size)]
            legal_actions = legal_actions[torch.tensor(maintain).flatten(), :]
        if pause_action:
            legal_actions = legal_actions.reshape(states.batch_size, -1, 2)
            legal_actions = torch.cat([legal_actions, (legal_actions[:, 0] * 0).unsqueeze(1)], dim=1).view(-1, 2)

    return legal_actions


def step_batch(states, action, action_type='swap', return_sub_reward=False):
    """
    :param states: LightGraph
    :param action: torch.tensor((batch_size, 2))
    :return:
    """
    assert states.batch_size == action.shape[0]

    if states.in_cuda:
        mask = torch.tensor(range(0, states.n * states.batch_size, states.n)).cuda()
    else:
        mask = torch.tensor(range(0, states.n * states.batch_size, states.n))

    batch_size = states.batch_size
    n = states.n

    ii, jj = action[:, 0], action[:, 1]

    old_S = states.kcut_value

    if action_type == 'swap':
        # swap two sets of nodes
        tmp = dc(states.ndata['label'][ii + mask])
        states.ndata['label'][ii + mask] = states.ndata['label'][jj + mask]
        states.ndata['label'][jj + mask] = tmp
    else:
        # flip nodes
        states.ndata['label'][ii + mask] = torch.nn.functional.one_hot(jj, states.k).float()
    # rewire edges
    nonzero_idx = [i for i in range(states.n ** 2) if i % (states.n + 1) != 0]
    states.edata['e_type'][:, 1] = torch.bmm(states.ndata['label'].view(batch_size, n, -1),
                                             states.ndata['label'].view(batch_size, n, -1).transpose(1, 2)) \
                                       .view(batch_size, -1)[:, nonzero_idx].view(-1)

    # compute new S
    new_S = calc_S(states)
    states.kcut_value = new_S

    rewards = old_S - new_S

    # if return_sub_reward:
    #     # compute old S
    #     #  (b, k, n*(n-1))
    #     group_matrix_k = torch.bmm(
    #         states.ndata['label'].view(batch_size, n, -1).transpose(1, 2).reshape(batch_size * self.k, n, 1),
    #         states.ndata['label'].view(batch_size, n, -1).transpose(1, 2).reshape(batch_size * self.k, n, 1).transpose(1, 2)).view(batch_size * self.k, -1)[:, self.graph_generator.nonzero_idx].view(batch_size, self.k, -1)
    #     #  (k, b*n*(n-1))
    #     remain_edges = group_matrix_k.transpose(0, 1).reshape(self.k, -1) * states.edata['d'][:, 0]
    #     #  (k, b)
    #     old_S_k = remain_edges.view(self.k, batch_size, -1).sum(dim=2)
    #     #  (b)
    #     old_S = old_S_k.sum(dim=0)
    #
    #     # swap two sets of nodes
    #     tmp = dc(states.ndata['label'][ii + self.mask1])
    #     states.ndata['label'][ii + self.mask1] = states.ndata['label'][jj + self.mask1]
    #     states.ndata['label'][jj + self.mask1] = tmp
    #
    #     # rewire edges
    #     states.edata['e_type'][:, 1] = torch.bmm(states.ndata['label'].view(batch_size, n, -1),
    #                                              states.ndata['label'].view(batch_size, n, -1).transpose(1,
    #                                                                                                      2)).view(
    #         batch_size, -1)[:, self.graph_generator.nonzero_idx].view(-1)
    #
    #     # compute new S
    #     #  (b, k, n*(n-1))
    #     group_matrix_k = torch.bmm(
    #         states.ndata['label'].view(batch_size, n, -1).transpose(1, 2).reshape(batch_size * self.k, n, 1),
    #         states.ndata['label'].view(batch_size, n, -1).transpose(1, 2).reshape(batch_size * self.k, n, 1).transpose(1, 2)).view(batch_size * self.k, -1)[:, self.graph_generator.nonzero_idx].view(batch_size, self.k, -1)
    #     #  (k, b*n*(n-1))
    #     remain_edges = group_matrix_k.transpose(0, 1).reshape(self.k, -1) * states.edata['d'][:, 0]
    #     #  (k, b)
    #     new_S_k = remain_edges.view(self.k, batch_size, -1).sum(dim=2)
    #     #  (b)
    #     new_S = new_S_k.sum(dim=0)
    #
    #     rewards = (old_S - new_S) / 2
    #
    #     sub_rewards = (old_S_k - new_S_k) / 2

    return states, rewards


# G = GraphGenerator(3,3,8, style='plain')
# g1 = G.generate_graph(batch_size=1, cuda_flag=False)
# g2 = G.generate_graph(batch_size=1, cuda_flag=False)
# gg = make_batch([g1, g2])
# a = get_legal_actions(gg, action_type='swap')
# peek_greedy_reward(states=gg, actions=a[:2,:])
# step_batch(gg, a[:2,:])


