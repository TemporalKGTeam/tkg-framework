import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from enum import Enum
from collections import defaultdict
from typing import Mapping, Dict

from tkge.common.registry import Registrable
from tkge.common.config import Config
from tkge.common.error import ConfigurationError
from tkge.data.dataset import Dataset


class BaseModel(nn.Module, Registrable):
    def __init__(self, config: Config, dataset: Dataset, device: str):
        nn.Module.__init__(self)
        Registrable.__init__(self, config=config)

        self.dataset = dataset
        self.device = device
        self.prepare_embedding()

    @staticmethod
    def create(config: Config, dataset: Dataset, device: str):
        """Factory method for sampler creation"""

        model_type = config.get("model.name")

        if model_type in BaseModel.list_available():
            kwargs = config.get("model.args")  # TODO: 需要改成key的格式
            return BaseModel.by_name(model_type)(config, dataset, device)
        else:
            raise ConfigurationError(
                f"{model_type} specified in configuration file is not supported"
                f"implement your model class with `BaseModel.register(name)"
            )

    def load_config(self):
        # TODO(gengyuan): 有参数的话加载，没指定参数的话用默认，最好可以直接读config文件然后setattr，需不需要做assert？
        raise NotImplementedError

    def prepare_embedding(self):
        raise NotImplementedError

    def get_embedding(self, **kwargs):
        raise NotImplementedError

    def forward(self, **kwargs):
        raise NotImplementedError


@BaseModel.register(name='de_simple')
class DeSimplEModel(BaseModel):
    def __init__(self, config: Config, dataset: Dataset, device: str = 'cpu'):
        super().__init__(config, dataset, device)

        self.time_nl = torch.sin  # TODO add to configuration file

    def prepare_embedding(self):
        num_ent = self.dataset.num_entities()
        num_rel = self.dataset.num_relations()

        s_emb_dim = self.config.get("model.embedding.s_emb_dim")
        t_emb_dim = self.config.get("model.embedding.t_emb_dim")

        device = self.device

        self.embedding: Dict[str, nn.Module] = defaultdict(dict)

        self.embedding.update({'ent_embs_h': nn.Embedding(num_ent, s_emb_dim).to(device)})
        self.embedding.update({'ent_embs_t': nn.Embedding(num_ent, s_emb_dim).to(device)})
        self.embedding.update({'rel_embs_f': nn.Embedding(num_rel, s_emb_dim + t_emb_dim).to(device)})
        self.embedding.update({'rel_embs_i': nn.Embedding(num_rel, s_emb_dim + t_emb_dim).to(device)})

        # frequency embeddings for the entities

        self.embedding.update({'m_freq_h': nn.Embedding(num_ent, t_emb_dim).to(device)})
        self.embedding.update({'m_freq_t': nn.Embedding(num_ent, t_emb_dim).to(device)})
        self.embedding.update({'d_freq_h': nn.Embedding(num_ent, t_emb_dim).to(device)})
        self.embedding.update({'d_freq_t': nn.Embedding(num_ent, t_emb_dim).to(device)})
        self.embedding.update({'y_freq_h': nn.Embedding(num_ent, t_emb_dim).to(device)})
        self.embedding.update({'y_freq_t': nn.Embedding(num_ent, t_emb_dim).to(device)})

        # phi embeddings for the entities
        self.embedding.update({'m_phi_h': nn.Embedding(num_ent, t_emb_dim).to(device)})
        self.embedding.update({'m_phi_t': nn.Embedding(num_ent, t_emb_dim).to(device)})
        self.embedding.update({'d_phi_h': nn.Embedding(num_ent, t_emb_dim).to(device)})
        self.embedding.update({'d_phi_t': nn.Embedding(num_ent, t_emb_dim).to(device)})
        self.embedding.update({'y_phi_h': nn.Embedding(num_ent, t_emb_dim).to(device)})
        self.embedding.update({'y_phi_t': nn.Embedding(num_ent, t_emb_dim).to(device)})

        # frequency embeddings for the entities
        self.embedding.update({'m_amps_h': nn.Embedding(num_ent, t_emb_dim).to(device)})
        self.embedding.update({'m_amps_t': nn.Embedding(num_ent, t_emb_dim).to(device)})
        self.embedding.update({'d_amps_h': nn.Embedding(num_ent, t_emb_dim).to(device)})
        self.embedding.update({'d_amps_t': nn.Embedding(num_ent, t_emb_dim).to(device)})
        self.embedding.update({'y_amps_h': nn.Embedding(num_ent, t_emb_dim).to(device)})
        self.embedding.update({'y_amps_t': nn.Embedding(num_ent, t_emb_dim).to(device)})

        self.embedding = nn.ModuleDict(self.embedding)

        for k, v in self.embedding.items():
            nn.init.xavier_uniform_(v.weight)

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

    def forward(self, sample):
        head = sample[:, 0]
        rel = sample[:, 1]
        tail = sample[:, 2]
        year = sample[:, 3].float()
        month = sample[:, 4].float()
        day = sample[:, 5].float()

        h_emb1, r_emb1, t_emb1, h_emb2, r_emb2, t_emb2 = self.get_embedding(head, rel, tail, year, month, day)

        p = self.config.get('model.dropout')

        score = ((h_emb1 * r_emb1) * t_emb1 + (h_emb2 * r_emb2) * t_emb2) / 2.0
        score = F.dropout(score, p=p, training=self.training)  # TODO training
        score = torch.sum(score, dim=1)

        return score, None


@BaseModel.register(name="tcomplex")
class TComplExModel(BaseModel):
    def __init__(self, config: Config, dataset: Dataset, device: str = 'cpu'):
        super().__init__(config, dataset, device)

        self.sizes = self.config.get("model.sizes")
        self.rank = self.config.get("model.get")
        self.no_time_emb = self.config.get("model.no_time_emb")
        self.init_size = self.config.get("model.init_size")

    def prepare_embedding(self):
        self.embedding = nn.ModuleList([
            nn.Embedding(s, 2 * self.rank, sparse=True)
            for s in [self.sizes[0], self.sizes[1], self.sizes[3]]
        ])

        for emb in self.embedding:
            emb.weight.data *= self.init_size

    def forward(self, x):
        """
        x is spot
        """
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight  # all ent tensor
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        # 1st item: scores
        # 2nd item: reg item factors
        # 3rd item: time

        score = (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() + \
                (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
        factor = {
            "n3_regularize": (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                              torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                              torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)),
            "lambda3_regularize": self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        }

        return score, factor

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


@BaseModel.register(name="hyte")
class HyTEModel(BaseModel):
    def __init__(self, config: Config, dataset: Dataset, device: str = 'cpu'):
        super().__init__(config, dataset, device)


@BaseModel.register(name="atise")
class ATiSEModel(BaseModel):
    def __init__(self, config: Config, dataset: Dataset, device: str = 'cpu'):
        super().__init__(config, dataset, device)

        self.gamma = self.config.get("model.gamma")
        self.cmin = self.config.get("model.cmin")
        self.cmax = self.config.get("model.cmax")
        self.emb_dim = self.config.get("model.embedding_dim")

    def prepare_embedding(self):
        num_ent = self.dataset.num_entities()
        num_rel = self.dataset.num_relations()

        device = self.device

        self.embedding: Dict[str, nn.Module] = defaultdict(None)

        self.embedding.update({'emb_E': nn.Embedding(num_ent, self.emb_dim, padding_idx=0).to(device)})
        self.embedding.update({'emb_E_var': nn.Embedding(num_ent, self.emb_dim, padding_idx=0).to(device)})
        self.embedding.update({'emb_R': nn.Embedding(num_rel, self.emb_dim, padding_idx=0).to(device)})
        self.embedding.update({'emb_R_var': nn.Embedding(num_rel, self.emb_dim, padding_idx=0).to(device)})

        self.embedding.update({'emb_TE': nn.Embedding(num_ent, self.emb_dim, padding_idx=0).to(device)})
        self.embedding.update({'alpha_E': nn.Embedding(num_ent, 1, padding_idx=0).to(device)})
        self.embedding.update({'beta_E': nn.Embedding(num_ent, self.emb_dim, padding_idx=0).to(device)})
        self.embedding.update({'omega_E': nn.Embedding(num_ent, self.emb_dim, padding_idx=0).to(device)})

        self.embedding.update({'emb_TR': nn.Embedding(num_rel, self.emb_dim, padding_idx=0).to(device)})
        self.embedding.update({'alpha_R': nn.Embedding(num_rel, 1, padding_idx=0).to(device)})
        self.embedding.update({'beta_R': nn.Embedding(num_rel, self.emb_dim, padding_idx=0).to(device)})
        self.embedding.update({'omega_R': nn.Embedding(num_rel, self.emb_dim, padding_idx=0).to(device)})

        self.embedding = nn.ModuleDict(self.embedding)

        r = 6 / np.sqrt(self.emb_dim)
        self.embedding['emb_E'].weight.data.uniform_(-r, r)
        self.embedding['emb_E_var'].weight.data.uniform_(self.cmin, self.cmax)
        self.embedding['emb_R'].weight.data.uniform_(-r, r)
        self.embedding['emb_R_var'].weight.data.uniform_(self.cmin, self.cmax)
        self.embedding['emb_TE'].weight.data.uniform_(-r, r)
        self.embedding['alpha_E'].weight.data.uniform_(0, 0)
        self.embedding['beta_E'].weight.data.uniform_(0, 0)
        self.embedding['omega_E'].weight.data.uniform_(-r, r)
        self.embedding['emb_TR'].weight.data.uniform_(-r, r)
        self.embedding['alpha_R'].weight.data.uniform_(0, 0)
        self.embedding['beta_R'].weight.data.uniform_(0, 0)
        self.embedding['omega_R'].weight.data.uniform_(-r, r)

        self.embedding['emb_E'].weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.embedding['emb_E_var'].weight.data.uniform_(self.cmin, self.cmax)
        self.embedding['emb_R'].weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.embedding['emb_R_var'].weight.data.uniform_(self.cmin, self.cmax)
        self.embedding['emb_TE'].weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.embedding['emb_TR'].weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def forward(self, x: torch.Tensor):
        # TODO(gengyuan) type conversion when feeding the data instead of running the models
        h_i, t_i, r_i, d_i = x[:, 0].long(), x[:, 1].long(), x[:, 2].long(), x[:, 3]

        pi = 3.14159265358979323846
        h_mean = self.embedding['emb_E'](h_i).view(-1, self.emb_dim) + \
                 d_i.view(-1, 1) * self.embedding['alpha_E'](h_i).view(-1, 1) * self.embedding['emb_TE'](h_i).view(-1,
                                                                                                                   self.emb_dim) \
                 + self.embedding['beta_E'](h_i).view(-1, self.emb_dim) * torch.sin(
            2 * pi * self.embedding['omega_E'](h_i).view(-1, self.emb_dim) * d_i.view(-1, 1))

        t_mean = self.embedding['emb_E'](t_i).view(-1, self.emb_dim) + \
                 d_i.view(-1, 1) * self.embedding['alpha_E'](t_i).view(-1, 1) * self.embedding['emb_TE'](t_i).view(-1,
                                                                                                                   self.emb_dim) \
                 + self.embedding['beta_E'](t_i).view(-1, self.emb_dim) * torch.sin(
            2 * pi * self.embedding['omega_E'](t_i).view(-1, self.emb_dim) * d_i.view(-1, 1))

        r_mean = self.embedding['emb_R'](r_i).view(-1, self.emb_dim) + \
                 d_i.view(-1, 1) * self.embedding['alpha_R'](r_i).view(-1, 1) * self.embedding['emb_TR'](r_i).view(-1,
                                                                                                                   self.emb_dim) \
                 + self.embedding['beta_R'](r_i).view(-1, self.emb_dim) * torch.sin(
            2 * pi * self.embedding['omega_R'](r_i).view(-1, self.emb_dim) * d_i.view(-1, 1))

        h_var = self.embedding['emb_E_var'](h_i).view(-1, self.emb_dim)
        t_var = self.embedding['mb_E_var'](t_i).view(-1, self.emb_dim)
        r_var = self.embedding['emb_R_var'](r_i).view(-1, self.emb_dim)

        out1 = torch.sum((h_var + t_var) / r_var, 1) + torch.sum(((r_mean - h_mean + t_mean) ** 2) / r_var,
                                                                 1) - self.emb_dim
        out2 = torch.sum(r_var / (h_var + t_var), 1) + torch.sum(((h_mean - t_mean - r_mean) ** 2) / (h_var + t_var),
                                                                 1) - self.emb_dim
        scores = (out1 + out2) / 4

        return scores, None


# reference: https://github.com/bsantraigi/TA_TransE/blob/master/model.py
@BaseModel.register(name="ta_transe")
class TATransEModel(BaseModel):
    def __init__(self, config: Config, dataset: Dataset, device: str = 'cpu'):
        super().__init__(config, dataset, device)

        self.emb_dim = self.config.get("model.embedding_dim")
        self.l1_flag = self.config.get("model.l1_flag")

    def prepare_embedding(self):
        num_ent = self.dataset.num_entities()
        num_rel = self.dataset.num_relations()
        num_tem = self.dataset.num_timestamps()

        self.embedding: Dict[str, torch.nn.Embedding] = defaultdict(None)
        self.embedding['ent'] = torch.nn.Embedding(num_ent, self.emb_dim)
        self.embedding['rel'] = torch.nn.Embedding(num_rel, self.emb_dim)
        self.embedding['tem'] = torch.nn.Embedding(num_tem, self.emb_dim)

        for _, emb in self.embedding.items():
            torch.nn.init.xavier_uniform_(emb)
            emb.weight.data.renorm(p=2, dim=1)

    def forward(self, x: torch.Tensor):
        h, r, t, t = x[:, 0].long(), x[:, 1].long(), x[:, 2].long(), x[:, 3].long()

        h_e = self.embedding['ent'][h]
        t_e = self.embedding['ent'][t]
        r_e = self.embedding['rel'][h]
        tem_e = self.embedding['tem'][h]

        if self.f1_flag:
            scores = torch.sum(torch.abs(h_e + r_e + tem_e - t_e), 1)
        else:
            scores = torch.sum((h_e + r_e + tem_e - t_e) ** 2, 1)

        return scores, None


# reference: https://github.com/bsantraigi/TA_TransE/blob/master/model.py
@BaseModel.register(name="ta_transe")
class TATransEModel(BaseModel):
    def __init__(self, config: Config, dataset: Dataset, device: str = 'cpu'):
        super().__init__(config, dataset, device)

        self.emb_dim = self.config.get("model.embedding_dim")
        self.l1_flag = self.config.get("model.l1_flag")

    def prepare_embedding(self):
        num_ent = self.dataset.num_entities()
        num_rel = self.dataset.num_relations()
        num_tem = self.dataset.num_timestamps()

        self.embedding: Dict[str, torch.nn.Embedding] = defaultdict(None)
        self.embedding['ent'] = torch.nn.Embedding(num_ent, self.emb_dim)
        self.embedding['rel'] = torch.nn.Embedding(num_rel, self.emb_dim)
        self.embedding['tem'] = torch.nn.Embedding(num_tem, self.emb_dim)

        for _, emb in self.embedding.items():
            torch.nn.init.xavier_uniform_(emb)
            emb.weight.data.renorm(p=2, dim=1)

    def forward(self, x: torch.Tensor):
        h, r, t, t = x[:, 0].long(), x[:, 1].long(), x[:, 2].long(), x[:, 3].long()

        h_e = self.embedding['ent'][h]
        t_e = self.embedding['ent'][t]
        r_e = self.embedding['rel'][h]
        tem_e = self.embedding['tem'][h]

        if self.f1_flag:
            scores = torch.sum(torch.abs(h_e + r_e + tem_e - t_e), 1)
        else:
            scores = torch.sum((h_e + r_e + tem_e - t_e) ** 2, 1)

        return scores, None
