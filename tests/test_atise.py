import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
from torch.nn.init import xavier_normal_

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
from numpy.random import RandomState

from collections import defaultdict
import time

from tkge.data.dataset import SplitDataset
from tkge.data.custom_dataset import ICEWS14AtiseDatasetProcessor
from tkge.eval.metrics import Evaluation
from tkge.train.sampling import NonNegativeSampler

from Dataset import KnowledgeGraph

randseed = 9999
np.random.seed(randseed)
torch.manual_seed(randseed)


class MockAtiseDatasetProcessor(ICEWS14AtiseDatasetProcessor):
    def __init__(self):
        self.folder = "/home/gengyuan/workspace/tkge/data/icews14"
        self.level = "day"
        self.index = False
        self.float = True

        self.train_raw = []
        self.valid_raw = []
        self.test_raw = []

        self.reciprocal_training = True

        self.ent2id = dict()
        self.rel2id = dict()
        with open('/home/gengyuan/workspace/baseline/ATISE/icews14/entity2id.txt', 'r') as f:
            ent2id = f.readlines()
            for line in ent2id:
                line = line.split('\t')
                self.ent2id[line[0]] = int(line[1])

        with open('/home/gengyuan/workspace/baseline/ATISE/icews14/relation2id.txt', 'r') as f:
            rel2id = f.readlines()
            for line in rel2id:
                line = line.split('\t')
                self.rel2id[line[0]] = int(line[1])

        self.ts2id = defaultdict(None)

        self.train_set = defaultdict(list)
        self.valid_set = defaultdict(list)
        self.test_set = defaultdict(list)

        self.all_triples = []
        self.all_quadruples = []

        self.load()
        self.process()
        self.filter()

    def index_relations(self, rel: str):
        if rel.endswith('(RECIPROCAL)'):
            return self.rel2id[rel[:-12]] + 230
        else:
            return self.rel2id[rel]

    def index_entities(self, ent: str):
        if ent == 'Horacio González':
            ent = 'Horacio Gonzalez'
        return self.ent2id[ent]

    def process_time(self, origin: str):
        # TODO (gengyuan) move to init method
        self.gran = 3

        start_sec = time.mktime(time.strptime('2014-01-01', '%Y-%m-%d'))

        end_sec = time.mktime(time.strptime(origin, '%Y-%m-%d'))
        day = int((end_sec - start_sec) / (self.gran * 24 * 60 * 60))

        return day


class MockEvaluation(Evaluation):
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.vocab_size = dataset.num_entities()

        self.device = device
        self.filter = "time-aware"
        self.ordering = "optimistic"
        self.k = [1, 3, 10]

        self.filtered_data = defaultdict(None)
        self.filtered_data['sp_'] = self.dataset.filter(type=self.filter, target='o')
        self.filtered_data['_po'] = self.dataset.filter(type=self.filter, target='s')


class MockSampler(NonNegativeSampler):
    def __init__(self, dataset, as_matrix):
        self.filter = False
        self.as_matrix = as_matrix
        self.dataset = dataset


class ATISE(nn.Module):
    def __init__(self, n_entity, n_relation, embedding_dim, batch_size, learning_rate, gamma, cmin, cmax, gpu=True):
        super(ATISE, self).__init__()
        self.gpu = gpu
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.cmin = cmin
        self.cmax = cmax
        # Nets
        self.emb_E = torch.nn.Embedding(n_entity, self.embedding_dim, padding_idx=0)
        self.emb_E_var = torch.nn.Embedding(n_entity, self.embedding_dim, padding_idx=0)
        self.emb_R = torch.nn.Embedding(n_relation, self.embedding_dim, padding_idx=0)
        self.emb_R_var = torch.nn.Embedding(n_relation, self.embedding_dim, padding_idx=0)
        self.emb_TE = torch.nn.Embedding(n_entity, self.embedding_dim, padding_idx=0)
        self.alpha_E = torch.nn.Embedding(n_entity, 1, padding_idx=0)
        self.beta_E = torch.nn.Embedding(n_entity, self.embedding_dim, padding_idx=0)
        self.omega_E = torch.nn.Embedding(n_entity, self.embedding_dim, padding_idx=0)
        self.emb_TR = torch.nn.Embedding(n_relation, self.embedding_dim, padding_idx=0)
        self.alpha_R = torch.nn.Embedding(n_relation, 1, padding_idx=0)
        self.beta_R = torch.nn.Embedding(n_relation, self.embedding_dim, padding_idx=0)
        self.omega_R = torch.nn.Embedding(n_relation, self.embedding_dim, padding_idx=0)

        # Initialization
        r = 6 / np.sqrt(self.embedding_dim)
        self.emb_E.weight.data.uniform_(-r, r)
        self.emb_E_var.weight.data.uniform_(self.cmin, self.cmax)
        self.emb_R.weight.data.uniform_(-r, r)
        self.emb_R_var.weight.data.uniform_(self.cmin, self.cmax)
        self.emb_TE.weight.data.uniform_(-r, r)
        self.alpha_E.weight.data.uniform_(0, 0)
        self.beta_E.weight.data.uniform_(0, 0)
        self.omega_E.weight.data.uniform_(-r, r)
        self.emb_TR.weight.data.uniform_(-r, r)
        self.alpha_R.weight.data.uniform_(0, 0)
        self.beta_R.weight.data.uniform_(0, 0)
        self.omega_R.weight.data.uniform_(-r, r)

        # Regularization
        self.normalize_embeddings()

        if self.gpu:
            self.cuda()

    def forward(self, X):
        # h_i, t_i, r_i, d_i = X[:, 0].astype(np.int64), X[:, 1].astype(np.int64), X[:, 2].astype(np.int64), X[:,
        #                                                                                                    3].astype(
        #     np.float32)
        h_i, t_i, r_i, d_i = X[:, 0].long(), X[:, 2].long(), X[:, 1].long(), X[:, 3].float()

        # if self.gpu:
        #     h_i = Variable(torch.from_numpy(h_i).cuda())
        #     t_i = Variable(torch.from_numpy(t_i).cuda())
        #     r_i = Variable(torch.from_numpy(r_i).cuda())
        #     d_i = Variable(torch.from_numpy(d_i).cuda())
        #
        # else:
        #     h_i = Variable(torch.from_numpy(h_i))
        #     t_i = Variable(torch.from_numpy(t_i))
        #     r_i = Variable(torch.from_numpy(r_i))
        #     d_i = Variable(torch.from_numpy(d_i))

        pi = 3.14159265358979323846
        h_mean = self.emb_E(h_i).view(-1, self.embedding_dim) + \
                 d_i.view(-1, 1) * self.alpha_E(h_i).view(-1, 1) * self.emb_TE(h_i).view(-1, self.embedding_dim) \
                 + self.beta_E(h_i).view(-1, self.embedding_dim) * torch.sin(
            2 * pi * self.omega_E(h_i).view(-1, self.embedding_dim) * d_i.view(-1, 1))

        t_mean = self.emb_E(t_i).view(-1, self.embedding_dim) + \
                 d_i.view(-1, 1) * self.alpha_E(t_i).view(-1, 1) * self.emb_TE(t_i).view(-1, self.embedding_dim) \
                 + self.beta_E(t_i).view(-1, self.embedding_dim) * torch.sin(
            2 * pi * self.omega_E(t_i).view(-1, self.embedding_dim) * d_i.view(-1, 1))

        r_mean = self.emb_R(r_i).view(-1, self.embedding_dim) + \
                 d_i.view(-1, 1) * self.alpha_R(r_i).view(-1, 1) * self.emb_TR(r_i).view(-1, self.embedding_dim) \
                 + self.beta_R(r_i).view(-1, self.embedding_dim) * torch.sin(
            2 * pi * self.omega_R(r_i).view(-1, self.embedding_dim) * d_i.view(-1, 1))

        h_var = self.emb_E_var(h_i).view(-1, self.embedding_dim)
        t_var = self.emb_E_var(t_i).view(-1, self.embedding_dim)
        r_var = self.emb_R_var(r_i).view(-1, self.embedding_dim)

        out1 = torch.sum((h_var + t_var) / r_var, 1) + torch.sum(((r_mean - h_mean + t_mean) ** 2) / r_var,
                                                                 1) - self.embedding_dim
        out2 = torch.sum(r_var / (h_var + t_var), 1) + torch.sum(((h_mean - t_mean - r_mean) ** 2) / (h_var + t_var),
                                                                 1) - self.embedding_dim
        out = (out1 + out2) / 4

        return out

    def log_rank_loss(self, y_pos, y_neg, temp=0):
        M = y_pos.size(0)
        N = y_neg.size(0)
        y_pos = self.gamma - y_pos
        y_neg = self.gamma - y_neg
        C = int(N / M)
        y_neg = y_neg.view(C, -1).transpose(0, 1)
        p = F.softmax(temp * y_neg)
        loss_pos = torch.sum(F.softplus(-1 * y_pos))
        loss_neg = torch.sum(p * F.softplus(y_neg))
        loss = (loss_pos + loss_neg) / 2 / M
        if self.gpu:
            loss = loss.cuda()
        return loss

    def rank_loss(self, y_pos, y_neg):
        M = y_pos.size(0)
        N = y_neg.size(0)
        C = int(N / M)
        y_pos = y_pos.repeat(C)
        if self.gpu:
            target = Variable(torch.from_numpy(-np.ones(N, dtype=np.float32))).cuda()
        else:
            target = Variable(torch.from_numpy(-np.ones(N, dtype=np.float32))).cpu()
        loss = nn.MarginRankingLoss(margin=self.gamma)
        loss = loss(y_pos, y_neg, target)
        return loss

    def normalize_embeddings(self):
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_E_var.weight.data.uniform_(self.cmin, self.cmax)
        self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_R_var.weight.data.uniform_(self.cmin, self.cmax)
        self.emb_TE.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_TR.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def regularization_embeddings(self):
        lower = torch.tensor(self.cmin).float()
        upper = torch.tensor(self.cmax).float()
        if self.gpu:
            lower = lower.cuda()
            upper = upper.cuda()
        self.emb_E_var.weight.data = torch.where(self.emb_E_var.weight.data < self.cmin, lower,
                                                 self.emb_E_var.weight.data)
        self.emb_E_var.weight.data = torch.where(self.emb_E_var.weight.data > self.cmax, upper,
                                                 self.emb_E_var.weight.data)
        self.emb_R_var.weight.data = torch.where(self.emb_R_var.weight.data < self.cmin, lower,
                                                 self.emb_R_var.weight.data)
        self.emb_R_var.weight.data = torch.where(self.emb_R_var.weight.data > self.cmax, upper,
                                                 self.emb_R_var.weight.data)
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_TE.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_TR.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def rank_left(self, X, facts, kg, timedisc, rev_set=0):

        rank = []
        with torch.no_grad():
            if timedisc:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.n_entity, 4])
                    i_score = torch.zeros(self.n_entity)
                    if self.gpu:
                        i_score = i_score.cuda()
                    for time_index in [triple[3], triple[4]]:
                        for i in range(0, self.n_entity):
                            X_i[i, 0] = i
                            X_i[i, 1] = triple[1]
                            X_i[i, 2] = triple[2]
                            X_i[i, 3] = time_index
                        i_score = i_score + self.forward(X_i).view(-1)
                        if rev_set > 0:
                            X_rev = np.ones([self.n_entity, 4])
                            for i in range(0, self.n_entity):
                                X_rev[i, 0] = triple[1]
                                X_rev[i, 1] = i
                                X_rev[i, 2] = triple[2] + self.n_relation // 2
                                X_rev[i, 3] = time_index
                            i_score = i_score + self.forward(X_rev).view(-1)
                    filter_out = kg.to_skip_final['lhs'][(fact[1], fact[2], fact[3], fact[4])]
                    target = i_score[int(triple[0])].clone()
                    i_score[filter_out] = 1e6
                    rank_triple = torch.sum((i_score < target).float()).cpu().item() + 1
                    rank.append(rank_triple)

            else:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.n_entity, 4])
                    for i in range(0, self.n_entity):
                        X_i[i, 0] = i
                        X_i[i, 1] = triple[1]
                        X_i[i, 2] = triple[2]
                        X_i[i, 3] = triple[3]

                    i_score = self.forward(X_i)

                    if rev_set > 0:
                        X_rev = np.ones([self.n_entity, 4])
                        for i in range(0, self.n_entity):
                            X_rev[i, 0] = triple[1]
                            X_rev[i, 1] = i
                            X_rev[i, 2] = triple[2] + self.n_relation // 2
                            X_rev[i, 3] = triple[3]
                        i_score = i_score + self.forward(X_rev).view(-1)
                    if self.gpu:
                        i_score = i_score.cuda()
                    filter_out = kg.to_skip_final['lhs'][(fact[1], fact[2], fact[3], fact[4])]
                    target = i_score[int(triple[0])].clone()
                    i_score[filter_out] = 1e6
                    rank_triple = torch.sum((i_score < target).float()).cpu().item() + 1
                    rank.append(rank_triple)

                print('left')
                print(rank)
        return rank

    def rank_right(self, X, facts, kg, timedisc, rev_set=0):
        rank = []
        with torch.no_grad():
            if timedisc:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.n_entity, 4])
                    i_score = torch.zeros(self.n_entity)
                    if self.gpu:
                        i_score = i_score.cuda()
                    for time_index in [triple[3], triple[4]]:
                        for i in range(0, self.n_entity):
                            X_i[i, 0] = triple[0]
                            X_i[i, 1] = i
                            X_i[i, 2] = triple[2]
                            X_i[i, 3] = time_index
                        i_score = i_score + self.forward(X_i).view(-1)
                        if rev_set > 0:
                            X_rev = np.ones([self.n_entity, 4])
                            for i in range(0, self.n_entity):
                                X_rev[i, 0] = i
                                X_rev[i, 1] = triple[0]
                                X_rev[i, 2] = triple[2] + self.n_relation // 2
                                X_rev[i, 3] = time_index
                            i_score = i_score + self.forward(X_rev).view(-1)

                    filter_out = kg.to_skip_final['rhs'][(fact[0], fact[2], fact[3], fact[4])]
                    target = i_score[int(triple[1])].clone()
                    i_score[filter_out] = 1e6
                    rank_triple = torch.sum((i_score < target).float()).cpu().item() + 1

                    rank.append(rank_triple)
            else:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.n_entity, 4])
                    for i in range(0, self.n_entity):
                        X_i[i, 0] = triple[0]
                        X_i[i, 1] = i
                        X_i[i, 2] = triple[2]
                        X_i[i, 3] = triple[3]
                    i_score = self.forward(X_i)
                    if rev_set > 0:
                        X_rev = np.ones([self.n_entity, 4])
                        for i in range(0, self.n_entity):
                            X_rev[i, 0] = i
                            X_rev[i, 1] = triple[0]
                            X_rev[i, 2] = triple[2] + self.n_relation // 2
                            X_rev[i, 3] = triple[3]
                        i_score = i_score + self.forward(X_rev).view(-1)
                    if self.gpu:
                        i_score = i_score.cuda()
                    filter_out = kg.to_skip_final['rhs'][(fact[0], fact[2], fact[3], fact[4])]
                    target = i_score[int(triple[1])].clone()
                    i_score[filter_out] = 1e6
                    rank_triple = torch.sum((i_score < target).float()).cpu().item() + 1

                    rank.append(rank_triple)

                print('right')
                print(rank)

        return rank

    def timepred(self, X):
        rank = []
        with torch.no_grad():
            for triple in X:
                X_i = np.ones([self.n_day, len(triple)])
                for i in range(self.n_day):
                    X_i[i, 0] = triple[0]
                    X_i[i, 1] = triple[1]
                    X_i[i, 2] = triple[2]
                    X_i[i, 3:] = self.time_dict[i]
                i_score = self.forward(X_i)
                if self.gpu:
                    i_score = i_score.cuda()

                target = i_score[triple[3]]
                rank_triple = torch.sum((i_score < target).float()).cpu().item() + 1
                rank.append(rank_triple)

        return rank


model_path = "/home/gengyuan/workspace/baseline/ATISE/icews14/ATISE/timediscrete0/dim500/lr0.0000/neg_num10/3day/gamma120/cmin0.0030/params.pkl"

model = ATISE(7129, 460, 500, 64, 0, 120, 0.003, 0.3, True)
model_state_dict = torch.load(model_path)
model.load_state_dict(model_state_dict)


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = MockAtiseDatasetProcessor()
    evaluator = MockEvaluation(dataset, device)
    sampler = MockSampler(dataset, as_matrix=True)
    #
    # print(dataset.ts2id)
    # print(dataset.index_timestamps(23))
    # print(dataset.index_relations('Arrest, detain, or charge with legal action'))
    # print(dataset.index_entities('Japan'))
    # print(dataset.index_entities('Police (Japan)'))

    # print(dataset.filter(type='time-aware', target='s')['None-9-2205-33'])
    # print(dataset.get("train")['triple'][15442])
    # print(dataset.get("train")['timestamp_id'][15442])
    # print(dataset.get("train")['timestamp_float'][15442])
    #
    #
    # assert False

    valid_loader = torch.utils.data.DataLoader(
        SplitDataset(dataset.get("test"), ['timestamp_float', 'timestamp_id']),
        shuffle=False,
        batch_size=1,
        num_workers=0
    )

    with torch.no_grad():
        model.eval()

        metrics = dict()
        metrics['head'] = defaultdict(float)
        metrics['tail'] = defaultdict(float)

        rank_left = []
        rank_right = []
        scores_head = []
        scores_tail = []

        filter_left = []
        filter_right = []

        l = 0

        dfs = dataset.filter(type="time-aware", target="s")
        dfo = dataset.filter(type="time-aware", target="o")

        for batch in valid_loader:
            bs = batch.size(0)
            dim = batch.size(1)

            l += bs

            print(l)

            samples_head, _ = sampler.sample(batch, "head")
            samples_tail, _ = sampler.sample(batch, "tail")

            samples_head = samples_head.to(device)
            samples_tail = samples_tail.to(device)

            samples_head = samples_head.view(-1, dim)
            samples_tail = samples_tail.view(-1, dim)

            batch_scores_head = model.forward(samples_head)
            batch_scores_tail = model.forward(samples_tail)

            batch_scores_head = batch_scores_head.view(bs, -1)
            batch_scores_tail = batch_scores_tail.view(bs, -1)

            # reciprocal
            samples_head_reciprocal = samples_head.clone().view(-1, dim)
            samples_tail_reciprocal = samples_tail.clone().view(-1, dim)

            samples_head_reciprocal[:, 1] += 230
            samples_head_reciprocal[:, [0, 2]] = samples_head_reciprocal.index_select(1, torch.Tensor(
                [2, 0]).long().to(device))

            samples_tail_reciprocal[:, 1] += 230
            samples_tail_reciprocal[:, [0, 2]] = samples_tail_reciprocal.index_select(1, torch.Tensor(
                [2, 0]).long().to(device))

            batch_scores_head_reci = model.forward(samples_head_reciprocal)
            batch_scores_tail_reci = model.forward(samples_tail_reciprocal)

            batch_scores_head_reci = batch_scores_head_reci.view(bs, -1)
            batch_scores_tail_reci = batch_scores_tail_reci.view(bs, -1)

            batch_scores_head += batch_scores_head_reci
            batch_scores_tail += batch_scores_tail_reci

            scores_head.append(batch_scores_head)
            scores_tail.append(batch_scores_tail)

            batch_metrics = dict()

            batch_metrics['head'] = evaluator.eval(batch, batch_scores_head, miss='s')
            batch_metrics['tail'] = evaluator.eval(batch, batch_scores_tail, miss='o')

            # print filter
            filter_left.append(
                dfs[f'None-{int(batch[0, 1])}-{int(batch[0, 2])}-{int(batch[0, -1])}'])
            filter_right.append(
                dfo[f'{int(batch[0, 0])}-{int(batch[0, 1])}-None-{int(batch[0, -1])}'])

            # rank_left.append(batch_metrics['head']['mean_ranking'])
            # rank_right.append(batch_metrics['tail']['mean_ranking'])

            # TODO(gengyuan) refactor
            for pos in ['head', 'tail']:
                for key in batch_metrics[pos].keys():
                    metrics[pos][key] += batch_metrics[pos][key] * bs

        # rank = rank_left + rank_right
        # torch.save(rank, "/home/gengyuan/workspace/baseline/ATISE/rank_tkge.pt")
        # rank2 = torch.load("/home/gengyuan/workspace/baseline/ATISE/rank.pt")
        #
        # print('assert Equal')
        # print(rank==rank2)

        # torch.save(scores_head + scores_tail, "/home/gengyuan/workspace/baseline/ATISE/scores_tkge.pt")
        torch.save(filter_left, "/home/gengyuan/workspace/baseline/ATISE/filter_left_tkge.pt")
        torch.save(filter_right, "/home/gengyuan/workspace/baseline/ATISE/filter_right_tkge.pt")

        for pos in ['head', 'tail']:
            for key in metrics[pos].keys():
                metrics[pos][key] /= l

        print(f"Metrics(head prediction) in iteration : {metrics['head'].items()}")
        print(f"Metrics(tail prediction) in iteration : {metrics['tail'].items()}")


def test_sc():
    def mean_rank(rank):
        m_r = 0
        N = len(rank)
        for i in rank:
            m_r = m_r + i / N

        return m_r

    def mrr(rank):
        mrr = 0
        N = len(rank)
        for i in rank:
            mrr = mrr + 1 / i / N

        return mrr

    def hit_N(rank, N):
        hit = 0
        for i in rank:
            if i <= N:
                hit = hit + 1

        hit = hit / len(rank)

        return hit

    kg = KnowledgeGraph(data_dir="/home/gengyuan/workspace/baseline/ATISE/icews14", gran=3, rev_set=1)

    test_pos = np.array(kg.test_triples)
    print(test_pos)

    rank = model.rank_left(test_pos, kg.test_facts, kg, 0, rev_set=1)
    rank_right = model.rank_right(test_pos, kg.test_facts, kg, 0, rev_set=1)
    rank = rank + rank_right

    m_rank = mean_rank(rank)
    mean_rr = mrr(rank)
    hit_1 = hit_N(rank, 1)
    hit_3 = hit_N(rank, 3)
    hit_5 = hit_N(rank, 5)
    hit_10 = hit_N(rank, 10)
    print('validation results:')
    print('Mean Rank: {:.0f}'.format(m_rank))
    print('Mean RR: {:.4f}'.format(mean_rr))
    print('Hit@1: {:.4f}'.format(hit_1))
    print('Hit@3: {:.4f}'.format(hit_3))
    print('Hit@5: {:.4f}'.format(hit_5))
    print('Hit@10: {:.4f}'.format(hit_10))


if __name__ == '__main__':
    test()
