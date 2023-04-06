import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(".")


class HMCN(nn.Module):
    def __init__(self, input_dim, voc_size):
        super(HMCN, self).__init__()
        self.input_dim = input_dim
        self.voc_size = voc_size
        ## Local hierarchical classifiers
        self.lhc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, 2*input_dim),
            nn.ReLU(),
            nn.Linear(2*input_dim, sum(voc_size)),
        )
        ## Global hierarchical classifiers
        self.ghc = nn.GRU(input_dim, input_dim, batch_first=True)
        self.alpha = nn.Parameter(torch.FloatTensor([[0.]*sum(voc_size)]))
        self.local_out = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(input_dim, voc_size[i])
                )
                for i in range(len(voc_size))]
        )

    def forward(self, x):
        ## Local output
        feat1 = self.lhc(x)
        p1 = torch.sigmoid(feat1)
        ## Global output
        x = x.unsqueeze(1)
        seq_x = torch.cat([x, x, x], dim=1)
        seq_o, _ = self.ghc(seq_x)          # (1, seq, dim)
        feat2 = []
        for i, _ in enumerate(self.local_out):
            feat2.append(self.local_out[i](seq_o[:, i, :]))

        feat2 = torch.cat(feat2, dim=-1)    # (1, o_n)
        p2 = torch.sigmoid(feat2)
        p_weights = torch.sigmoid(self.alpha)

        return p1 * p_weights + (1-p_weights) * p2


class RETAIN(nn.Module):
    def __init__(self, emb_dim):
        super(RETAIN, self).__init__()
        self.emb_dim = emb_dim
        self.encoder = nn.GRU(emb_dim, emb_dim * 2, batch_first=True)
        self.alpha_net = nn.Linear(emb_dim * 2, 1)
        self.beta_net = nn.Linear(emb_dim * 2, emb_dim)

    def forward(self, i1_seq):
        o1, h1 = self.encoder(i1_seq)                     # o1:(1, seq, dim*2) hi:(1,1,dim*2)

        ej1 = self.alpha_net(o1)                          # (1, seq, 1)
        bj1 = self.beta_net(o1)                           # (1, seq, dim)
        att_ej1 = torch.softmax(ej1, dim=1)
        o1 = (att_ej1 * torch.tanh(bj1) * i1_seq).sum(1)  # (1, dim)

        return o1


class CEHMRNet(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_dim=64,
                 device=torch.device('cpu:0'),
                 ):
        r"""
        The designed network in CEHMRNet framework.
        :param vocab_size: list, including sizes of diagnosis, procedures and medications.
        :param emb_dim:    embedding dimension
        :param device:      the device where program runs
        """
        super(CEHMRNet, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        # Embedding layers
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(2)])

        # Temporal representation
        self.retain = nn.ModuleList([RETAIN(emb_dim) for _ in range(2)])
        # Hierarchical classifiers
        voc_size = [vocab_size[4], vocab_size[3], vocab_size[2]]
        self.hmnc_f = HMCN(64*4, voc_size)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, input):

        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        if len(input) == 0:
            raise Exception("Input error!")

        i1_seq = []
        i2_seq = []
        for adm in input:
            i1_emb = self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))
            i2_emb = self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)))
            i1_seq.append(mean_embedding(i1_emb))
            i2_seq.append(mean_embedding(i2_emb))
        i1_seq = torch.cat(list(reversed(i1_seq)), dim=1)  # (1, seq, dim)
        i2_seq = torch.cat(list(reversed(i2_seq)), dim=1)  # (1, seq, dim)

        o1 = self.retain[0](i1_seq)
        o2 = self.retain[1](i2_seq)

        patient_representations = torch.cat([o1, o2], dim=-1)  # (seq, dim*2)

        query = patient_representations

        feat = torch.cat([i1_emb.squeeze(0).mean(0).unsqueeze(0), i2_emb.squeeze(0).mean(0).unsqueeze(0), query], dim=-1)
        output = self.hmnc_f(feat)

        return output

