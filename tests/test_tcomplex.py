import torch
from torch import nn
import torch.functional as F
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/desimple_master'
print(BASE_DIR)
sys.path.append(BASE_DIR)

from collections import defaultdict
from typing import Tuple

from tkge.data.dataset import ICEWS14DatasetProcessor, SplitDataset
from tkge.eval.metrics import Evaluation
from tkge.train.sampling import NonNegativeSampler, PseudoNegativeSampling


class MockICEWS14DatasetProcessor(ICEWS14DatasetProcessor):
    def __init__(self):
        self.folder = "/mnt/data1/ma/gengyuan/tkge/data/icews14"
        self.level = "day"
        self.index = True
        self.float = False

        self.reciprocal_training = True

        self.train_raw = []
        self.valid_raw = []
        self.test_raw = []

        mapping = torch.load('/mnt/data1/ma/gengyuan/baseline/tkbc/tkbc/mapping.pt')

        self.ent2id = mapping[0]
        self.rel2id = mapping[1]
        self.ts2id = mapping[2]

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


class Model(nn.Module):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(Model, self).__init__()

        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.no_time_emb = no_time_emb

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
             lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1]) * rhs[0] +
            (lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
             lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]  # rel @ time

        return (
                       (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
                       (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()  # 内积
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        return torch.cat([
            lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
            lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1],
            lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
            lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]
        ], 1)


class MockSampler(PseudoNegativeSampling):
    def __init__(self, dataset, as_matrix):
        self.filter = False
        self.as_matrix = as_matrix
        self.dataset = dataset


def test_sc():
    model_state_dict = torch.load('/mnt/data1/ma/gengyuan/baseline/tkbc/real_tcomplex_epoch49.pt')

    model = Model([7128, 460, 0, 365], rank=156)
    model.load_state_dict(model_state_dict)

    dataset = MockICEWS14DatasetProcessor()
    device = 'cpu'
    evaluator = MockEvaluation(dataset, device)

    print('==============================================')
    print(f"Number of entities : {dataset.num_entities()}")
    print(f"Number of relations : {dataset.num_relations()}")
    print(f"Number of relations : {dataset.num_timestamps()}")
    print(f"\n")
    print(f"Train set size : {dataset.train_size}")
    print(f"Valid set size : {dataset.valid_size}")
    print(f"Test set size : {dataset.test_size}")
    print('==============================================')

    valid_loader = torch.utils.data.DataLoader(
        SplitDataset(dataset.get("test"), ['timestamp_id']),
        shuffle=False,
        batch_size=1,
        num_workers=0
    )

    model.to(device)

    with torch.no_grad():
        model.eval()

        l = 0

        metrics = dict()
        metrics['head'] = defaultdict(float)
        metrics['tail'] = defaultdict(float)

        for batch in valid_loader:
            bs = batch.size(0)
            dim = batch.size(1)

            batch = batch.long().to(device)

            l += bs

            queries_head = batch.clone()
            queries_tail = batch.clone()

            # samples_head, _ = self.onevsall_sampler.sample(queries, "head")
            # samples_tail, _ = self.onevsall_sampler.sample(queries, "tail")

            # samples_head = samples_head.to(self.device)
            # samples_tail = samples_tail.to(self.device)

            queries_head[:, (0, 2)] = queries_head[:, (2, 0)]
            queries_head[:, 1]  = queries_head[:, 1]+ 230
            queries_tail[:, 1] = queries_tail[:, 1]

            batch_scores_head = model(queries_head)
            batch_scores_tail = model(queries_tail)

            batch_scores_head = batch_scores_head[0]
            batch_scores_tail = batch_scores_tail[0]

            # TODO(gengyuan) : 无论如何都要转化成matrix才可以计算evaluation

            # TODO (gengyuan): ATISE的eval可以统一进predict里面

            # if self.config.get("task.reciprocal_relation"):
            #     samples_head_reciprocal = samples_head.clone().view(-1, dim)
            #     samples_tail_reciprocal = samples_tail.clone().view(-1, dim)
            #
            #     samples_head_reciprocal[:, 1] += 1
            #     samples_head_reciprocal[:, [0, 2]] = samples_head_reciprocal.index_select(1, torch.Tensor(
            #         [2, 0]).long().to(self.device))
            #
            #     samples_tail_reciprocal[:, 1] += 1
            #     samples_tail_reciprocal[:, [0, 2]] = samples_tail_reciprocal.index_select(1, torch.Tensor(
            #         [2, 0]).long().to(self.device))
            #
            #     samples_head_reciprocal = samples_head_reciprocal.view(bs, -1)
            #     samples_tail_reciprocal = samples_tail_reciprocal.view(bs, -1)
            #
            #     batch_scores_head_reci, _ = self.model.predict(samples_head_reciprocal)
            #     batch_scores_tail_reci, _ = self.model.predict(samples_tail_reciprocal)
            #
            #     batch_scores_head += batch_scores_head_reci
            #     batch_scores_tail += batch_scores_tail_reci

            batch_metrics = dict()

            batch_metrics['head'] = evaluator.eval(batch, batch_scores_head, miss='s')
            batch_metrics['tail'] = evaluator.eval(batch, batch_scores_tail, miss='o')

            # TODO(gengyuan) refactor
            for pos in ['head', 'tail']:
                for key in batch_metrics[pos].keys():
                    metrics[pos][key] += batch_metrics[pos][key] * bs

        for pos in ['head', 'tail']:
            for key in metrics[pos].keys():
                metrics[pos][key] /= l

        print(f"Metrics(head prediction) in iteration: {metrics['head'].items()}")
        print(f"Metrics(tail prediction) in iteration: {metrics['tail'].items()}")


if __name__ == '__main__':
    test_sc()
