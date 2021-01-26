import torch
from torch import nn
import torch.functional as F
import os
import sys
import numpy as np
import random


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/desimple_master'
print(BASE_DIR)
sys.path.append(BASE_DIR)

from collections import defaultdict

from tkge.data.dataset import ICEWS14DatasetProcessor, SplitDataset, DatasetProcessor
from tkge.eval.metrics import Evaluation
from tkge.train.sampling import NonNegativeSampler
from tkge.models.utils import *

from desimple_master.tester import Tester
from desimple_master.dataset import Dataset


class MockICEWS14DatasetProcessor(ICEWS14DatasetProcessor):
    def __init__(self):
        self.folder = "/home/gengyuan/workspace/tkge/data/icews14"
        self.level = "day"
        self.index = False
        self.float = True

        self.reciprocal_training = False


        self.train_raw = []
        self.valid_raw = []
        self.test_raw = []

        self.ent2id = defaultdict(None)
        self.rel2id = defaultdict(None)
        self.ts2id = defaultdict(None)

        self.train_set = defaultdict(list)
        self.valid_set = defaultdict(list)
        self.test_set = defaultdict(list)

        self.all_triples = []
        self.all_quadruples = []

        self.load()
        self.process()
        self.filter()


class MockEvaluation(Evaluation):
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.vocab_size = dataset.num_entities()

        self.device = device
        self.filter = 'time-aware'
        self.preference = 'optimistic'
        self.ordering = 'descending'
        self.k = [1, 3, 10]

        self.filtered_data = defaultdict(None)
        self.filtered_data['sp_'] = self.dataset.filter(type=self.filter, target='o')
        self.filtered_data['_po'] = self.dataset.filter(type=self.filter, target='s')


class MockSampler(NonNegativeSampler):
    def __init__(self, dataset, as_matrix):
        self.filter = False
        self.as_matrix = as_matrix

        self.dataset = dataset


class MockDE_SimplE(torch.nn.Module):
    def __init__(self):
        super(MockDE_SimplE, self).__init__()

        self.ent_embs_h = nn.Embedding(7128, 68).cuda()
        self.ent_embs_t = nn.Embedding(7128, 68).cuda()
        self.rel_embs_f = nn.Embedding(230, 100).cuda()
        self.rel_embs_i = nn.Embedding(230, 100).cuda()

        self.create_time_embedds()

        self.time_nl = torch.sin

        nn.init.xavier_uniform_(self.ent_embs_h.weight)
        nn.init.xavier_uniform_(self.ent_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs_f.weight)
        nn.init.xavier_uniform_(self.rel_embs_i.weight)

    def create_time_embedds(self):

        # frequency embeddings for the entities
        self.m_freq_h = nn.Embedding(7128, 32).cuda()
        self.m_freq_t = nn.Embedding(7128, 32).cuda()
        self.d_freq_h = nn.Embedding(7128, 32).cuda()
        self.d_freq_t = nn.Embedding(7128, 32).cuda()
        self.y_freq_h = nn.Embedding(7128, 32).cuda()
        self.y_freq_t = nn.Embedding(7128, 32).cuda()

        # phi embeddings for the entities
        self.m_phi_h = nn.Embedding(7128, 32).cuda()
        self.m_phi_t = nn.Embedding(7128, 32).cuda()
        self.d_phi_h = nn.Embedding(7128, 32).cuda()
        self.d_phi_t = nn.Embedding(7128, 32).cuda()
        self.y_phi_h = nn.Embedding(7128, 32).cuda()
        self.y_phi_t = nn.Embedding(7128, 32).cuda()

        # frequency embeddings for the entities
        self.m_amps_h = nn.Embedding(7128, 32).cuda()
        self.m_amps_t = nn.Embedding(7128, 32).cuda()
        self.d_amps_h = nn.Embedding(7128, 32).cuda()
        self.d_amps_t = nn.Embedding(7128, 32).cuda()
        self.y_amps_h = nn.Embedding(7128, 32).cuda()
        self.y_amps_t = nn.Embedding(7128, 32).cuda()

        nn.init.xavier_uniform_(self.m_freq_h.weight)
        nn.init.xavier_uniform_(self.d_freq_h.weight)
        nn.init.xavier_uniform_(self.y_freq_h.weight)
        nn.init.xavier_uniform_(self.m_freq_t.weight)
        nn.init.xavier_uniform_(self.d_freq_t.weight)
        nn.init.xavier_uniform_(self.y_freq_t.weight)

        nn.init.xavier_uniform_(self.m_phi_h.weight)
        nn.init.xavier_uniform_(self.d_phi_h.weight)
        nn.init.xavier_uniform_(self.y_phi_h.weight)
        nn.init.xavier_uniform_(self.m_phi_t.weight)
        nn.init.xavier_uniform_(self.d_phi_t.weight)
        nn.init.xavier_uniform_(self.y_phi_t.weight)

        nn.init.xavier_uniform_(self.m_amps_h.weight)
        nn.init.xavier_uniform_(self.d_amps_h.weight)
        nn.init.xavier_uniform_(self.y_amps_h.weight)
        nn.init.xavier_uniform_(self.m_amps_t.weight)
        nn.init.xavier_uniform_(self.d_amps_t.weight)
        nn.init.xavier_uniform_(self.y_amps_t.weight)

    def get_time_embedd(self, entities, years, months, days, h_or_t):
        if h_or_t == "head":
            emb = self.y_amps_h(entities) * self.time_nl(
                self.y_freq_h(entities) * years + self.y_phi_h(entities))
            emb += self.m_amps_h(entities) * self.time_nl(
                self.m_freq_h(entities) * months + self.m_phi_h(entities))
            emb += self.d_amps_h(entities) * self.time_nl(
                self.d_freq_h(entities) * days + self.d_phi_h(entities))
        else:
            emb = self.y_amps_t(entities) * self.time_nl(
                self.y_freq_t(entities) * years + self.y_phi_t(entities))
            emb += self.m_amps_t(entities) * self.time_nl(
                self.m_freq_t(entities) * months + self.m_phi_t(entities))
            emb += self.d_amps_t(entities) * self.time_nl(
                self.d_freq_t(entities) * days + self.d_phi_t(entities))

        return emb

    def getEmbeddings(self, heads, rels, tails, years, months, days, intervals=None):
        years = years.view(-1, 1)
        months = months.view(-1, 1)
        days = days.view(-1, 1)
        h_embs1 = self.ent_embs_h(heads)
        r_embs1 = self.rel_embs_f(rels)
        t_embs1 = self.ent_embs_t(tails)

        h_embs2 = self.ent_embs_h(tails)
        r_embs2 = self.rel_embs_i(rels)
        t_embs2 = self.ent_embs_t(heads)

        h_embs1 = torch.cat((h_embs1, self.get_time_embedd(heads, years, months, days, "head")), 1)
        t_embs1 = torch.cat((t_embs1, self.get_time_embedd(tails, years, months, days, "tail")), 1)
        h_embs2 = torch.cat((h_embs2, self.get_time_embedd(tails, years, months, days, "head")), 1)
        t_embs2 = torch.cat((t_embs2, self.get_time_embedd(heads, years, months, days, "tail")), 1)

        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2

    def forward(self, heads, rels, tails, years, months, days):
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.getEmbeddings(heads, rels, tails, years,
                                                                                  months, days)
        scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        # scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = torch.sum(scores, dim=1)
        return scores

class DeSimplEModel(nn.Module):
    def __init__(self, dataset: DatasetProcessor):
        super().__init__()

        self.prepare_embedding()

        self.time_nl = torch.sin  # TODO add to configuration file

    def prepare_embedding(self):
        num_ent = self.dataset.num_entities()
        num_rel = self.dataset.num_relations()

        emb_dim = self.config.get("model.embedding.emb_dim")
        se_prop = self.config.get("model.embedding.se_prop")
        s_emb_dim = int(se_prop * emb_dim)
        t_emb_dim = emb_dim - s_emb_dim

        self.embedding = defaultdict(dict)

        self.embedding.update({'ent_embs_h': nn.Embedding(num_ent, s_emb_dim)})
        self.embedding.update({'ent_embs_t': nn.Embedding(num_ent, s_emb_dim)})
        self.embedding.update({'rel_embs_f': nn.Embedding(num_rel, s_emb_dim + t_emb_dim)})
        self.embedding.update({'rel_embs_i': nn.Embedding(num_rel, s_emb_dim + t_emb_dim)})

        # frequency embeddings for the entities

        self.embedding.update({'m_freq_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'m_freq_t': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'d_freq_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'d_freq_t': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'y_freq_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'y_freq_t': nn.Embedding(num_ent, t_emb_dim)})

        # phi embeddings for the entities
        self.embedding.update({'m_phi_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'m_phi_t': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'d_phi_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'d_phi_t': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'y_phi_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'y_phi_t': nn.Embedding(num_ent, t_emb_dim)})

        # frequency embeddings for the entities
        self.embedding.update({'m_amps_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'m_amps_t': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'d_amps_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'d_amps_t': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'y_amps_h': nn.Embedding(num_ent, t_emb_dim)})
        self.embedding.update({'y_amps_t': nn.Embedding(num_ent, t_emb_dim)})

        self.embedding = nn.ModuleDict(self.embedding)

        for k, v in self.embedding.items():
            nn.init.xavier_uniform_(v.weight)

        # torch.manual_seed(0)
        # torch.cuda.manual_seed_all(0)
        # np.random.seed(0)
        # random.seed(0)
        # torch.backends.cudnn.deterministic = True

        # nn.init.xavier_uniform_(self.embedding['ent_embs_h'].weight)
        # nn.init.xavier_uniform_(self.embedding['ent_embs_t'].weight)
        # nn.init.xavier_uniform_(self.embedding['rel_embs_f'].weight)
        # nn.init.xavier_uniform_(self.embedding['rel_embs_i'].weight)
        #
        # nn.init.xavier_uniform_(self.embedding['m_freq_h'].weight)
        # nn.init.xavier_uniform_(self.embedding['d_freq_h'].weight)
        # nn.init.xavier_uniform_(self.embedding['y_freq_h'].weight)
        # nn.init.xavier_uniform_(self.embedding['m_freq_t'].weight)
        # nn.init.xavier_uniform_(self.embedding['d_freq_t'].weight)
        # nn.init.xavier_uniform_(self.embedding['y_freq_t'].weight)
        #
        # nn.init.xavier_uniform_(self.embedding['m_phi_h'].weight)
        # nn.init.xavier_uniform_(self.embedding['d_phi_h'].weight)
        # nn.init.xavier_uniform_(self.embedding['y_phi_h'].weight)
        # nn.init.xavier_uniform_(self.embedding['m_phi_t'].weight)
        # nn.init.xavier_uniform_(self.embedding['d_phi_t'].weight)
        # nn.init.xavier_uniform_(self.embedding['y_phi_t'].weight)
        #
        # nn.init.xavier_uniform_(self.embedding['m_amps_h'].weight)
        # nn.init.xavier_uniform_(self.embedding['d_amps_h'].weight)
        # nn.init.xavier_uniform_(self.embedding['y_amps_h'].weight)
        # nn.init.xavier_uniform_(self.embedding['m_amps_t'].weight)
        # nn.init.xavier_uniform_(self.embedding['d_amps_t'].weight)
        # nn.init.xavier_uniform_(self.embedding['y_amps_t'].weight)

        # for name, params in self.named_parameters():
        #     print(name)
        #     print(params.weight)
        #
        # assert False

    def get_time_embedding(self, ent, year, month, day, ent_pos):
        # TODO: enum
        if ent_pos == "head":
            time_emb = self.embedding['y_amps_h'](ent) * self.time_nl(
                self.embedding['y_freq_h'](ent) * year + self.embedding['y_phi_h'](ent))
            time_emb += self.embedding['m_amps_h'](ent) * self.time_nl(
                self.embedding['m_freq_h'](ent) * month + self.embedding['m_phi_h'](ent))
            time_emb += self.embedding['d_amps_h'](ent) * self.time_nl(
                self.embedding['d_freq_h'](ent) * day + self.embedding['d_phi_h'](ent))
        else:
            time_emb = self.embedding['y_amps_t'](ent) * self.time_nl(
                self.embedding['y_freq_t'](ent) * year + self.embedding['y_phi_t'](ent))
            time_emb += self.embedding['m_amps_t'](ent) * self.time_nl(
                self.embedding['m_freq_t'](ent) * month + self.embedding['m_phi_t'](ent))
            time_emb += self.embedding['d_amps_t'](ent) * self.time_nl(
                self.embedding['d_freq_t'](ent) * day + self.embedding['d_phi_t'](ent))

        return time_emb

    def get_embedding(self, head, rel, tail, year, month, day):
        year = year.view(-1, 1)
        month = month.view(-1, 1)
        day = day.view(-1, 1)

        h_emb1 = self.embedding['ent_embs_h'](head)
        r_emb1 = self.embedding['rel_embs_f'](rel)
        t_emb1 = self.embedding['ent_embs_t'](tail)

        h_emb2 = self.embedding['ent_embs_t'](tail)
        r_emb2 = self.embedding['rel_embs_i'](rel)
        t_emb2 = self.embedding['ent_embs_h'](head)

        h_emb1 = torch.cat((h_emb1, self.get_time_embedding(head, year, month, day, 'head')), 1)
        t_emb1 = torch.cat((t_emb1, self.get_time_embedding(tail, year, month, day, 'tail')), 1)
        h_emb2 = torch.cat((h_emb2, self.get_time_embedding(tail, year, month, day, 'head')), 1)
        t_emb2 = torch.cat((t_emb2, self.get_time_embedding(head, year, month, day, 'tail')), 1)

        return h_emb1, r_emb1, t_emb1, h_emb2, r_emb2, t_emb2

    def forward(self, samples, **kwargs):
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()
        year = samples[:, 3]
        month = samples[:, 4]
        day = samples[:, 5]

        h_emb1, r_emb1, t_emb1, h_emb2, r_emb2, t_emb2 = self.get_embedding(head, rel, tail, year, month, day)

        p = self.config.get('model.dropout')

        scores = ((h_emb1 * r_emb1) * t_emb1 + (h_emb2 * r_emb2) * t_emb2) / 2.0
        scores = F.dropout(scores, p=p, training=self.training)  # TODO training
        scores = torch.sum(scores, dim=1)

        return scores, None

    def train_task(self, samples: torch.Tensor):
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factor = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factor

    def predict(self, queries: torch.Tensor):
        assert torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent."

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores


def test():
    model_path = "/home/gengyuan/workspace/tkge/tests/assets/500_512_0.001_0.0_68_500_0.4_32_20_0.68_500.chkpnt"
    model = torch.load(model_path, map_location='cuda')
    new_model = MockDE_SimplE()

    old_model_state = model.state_dict()
    old_model_state_keys = list(old_model_state.keys())
    for k in old_model_state_keys:
        print(k[7:])
        old_model_state[k[7:]] = old_model_state.pop(k)

    new_model.load_state_dict(old_model_state, strict=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = MockICEWS14DatasetProcessor()
    evaluator = MockEvaluation(dataset, device)
    sampler = MockSampler(dataset, as_matrix=True)

    print('==============================================')
    print(f"Number of entities : {dataset.num_entities()}")
    print(f"Number of relations : {dataset.num_relations()}")
    print(f"\n")
    print(f"Train set size : {dataset.train_size}")
    print(f"Valid set size : {dataset.valid_size}")
    print(f"Test set size : {dataset.test_size}")
    print('==============================================')

    valid_loader = torch.utils.data.DataLoader(
        SplitDataset(dataset.get("test"), ['timestamp_float', 'timestamp_id']),
        shuffle=False,
        batch_size=1,
        num_workers=0
    )


    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param)
    #     print(param.device)

    with torch.no_grad():
        model.eval()

        metrics = dict()
        metrics['head'] = defaultdict(float)
        metrics['tail'] = defaultdict(float)

        l = 0

        filt_list = {'head':[],
                     'tail':[]}
        rank_list = {'head':[],
                     'tail':[]}
        score_list = {'head':[],
                     'tail':[]}

        dfs = dataset.filter(type="time-aware", target="s")
        dfo = dataset.filter(type="time-aware", target="o")

        for batch in valid_loader:
            bs = batch.size(0)
            l += bs

            samples_head, _ = sampler.sample(batch, "head")
            samples_tail, _ = sampler.sample(batch, "tail")

            samples_head = samples_head.to(device)
            samples_tail = samples_tail.to(device)

            samples_head = samples_head.view(-1, 7)
            samples_tail = samples_tail.view(-1, 7)

            head = samples_head[:, 0].long()
            rel = samples_head[:, 1].long()
            tail = samples_head[:, 2].long()
            year = samples_head[:, 3]
            month = samples_head[:, 4]
            day = samples_head[:, 5]

            batch_scores_head = new_model(head, rel, tail, year, month, day)
            batch_scores_head = batch_scores_head.view(bs, -1)

            head = samples_tail[:, 0].long()
            rel = samples_tail[:, 1].long()
            tail = samples_tail[:, 2].long()
            year = samples_tail[:, 3]
            month = samples_tail[:, 4]
            day = samples_tail[:, 5]

            batch_scores_tail = new_model(head, rel, tail, year, month, day)
            batch_scores_tail = batch_scores_tail.view(bs, -1)

            # score_list['head'].append(batch_scores_head[0, batch[0, 0].long()].item())
            # score_list['tail'].append(batch_scores_tail[0, batch[0, 2].long()].item())
            #


            batch_metrics = dict()
            batch_metrics['head'] = evaluator.eval(batch, batch_scores_head, miss='s')
            batch_metrics['tail'] = evaluator.eval(batch, batch_scores_tail, miss='o')



            rank_list['head'].append(batch_metrics['head']['mean_ranking'])
            rank_list['tail'].append(batch_metrics['tail']['mean_ranking'])
            filt_list['head'].append(dfs[f'None-{int(batch[0, 1])}-{int(batch[0, 2])}-{int(batch[0, -1])}'])
            filt_list['tail'].append(dfo[f'{int(batch[0, 0])}-{int(batch[0, 1])}-None-{int(batch[0, -1])}'])

            for pos in ['head', 'tail']:
                for key in batch_metrics[pos].keys():
                    metrics[pos][key] += batch_metrics[pos][key] * bs

        torch.save(score_list, '/home/gengyuan/workspace/tkge/tests/desimple_master/scorelist_tkge.pt')
        # torch.save(filt_list, '/home/gengyuan/workspace/tkge/tests/desimple_master/filtlist_tkge.pt')

        for pos in ['head', 'tail']:
            for key in metrics[pos].keys():
                metrics[pos][key] /= l

        print(f"Metrics(head prediction)  : {metrics['head'].items()}")
        print(f"Metrics(tail prediction)  : {metrics['tail'].items()}")


def test_sc():
    model_path = "/home/gengyuan/workspace/tkge/tests/assets/500_512_0.001_0.0_68_500_0.4_32_20_0.68_500.chkpnt"
    model = torch.load(model_path, map_location='cuda')
    new_model = MockDE_SimplE()

    old_model_state = model.state_dict()
    old_model_state_keys = list(old_model_state.keys())
    for k in old_model_state_keys:
        print(k[7:])
        old_model_state[k[7:]] = old_model_state.pop(k)

    new_model.load_state_dict(old_model_state, strict=False)

    dataset = Dataset('icews14')
    tester = Tester(dataset, new_model, "test")
    tester.test()

def test_bp():
    input = torch.Tensor([])




if __name__ == '__main__':
    test()
