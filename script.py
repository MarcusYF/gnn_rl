device0 = torch.device("cuda:0")
model = DataParallel(alg, device_ids=[0, 1, 2, 3])
model.to(device0)

def to_cuda(G_, device):
    G = dc(G_)
    G.ndata['x'] = G.ndata['x'].cuda(device)
    G.ndata['label'] = G.ndata['label'].cuda(device)
    # G.ndata['h'] = G.ndata['h'].cuda(device)
    G.edata['d'] = G.edata['d'].cuda(device)
    G.edata['w'] = G.edata['w'].cuda(device)
    G.edata['e_type'] = G.edata['e_type'].cuda(device)
    return G
problem = KCut_DGL(k=5, m=6, adjacent_reserve=10, hidden_dim=32, sample_episode=10)
bg = problem.gen_batch_graph(batch_size=100)
batch_legal_actions = problem.get_legal_actions(state=bg)
S_a_encoding, h1, h2, Q_sa = model.forward(to_cuda(bg, device1), batch_legal_actions)