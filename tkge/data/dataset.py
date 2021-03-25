import torch
from torch.utils.data.dataset import Dataset as PTDataset

from typing import Dict, List, Optional
from collections import defaultdict

from tkge.common.registrable import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.common.error import ConfigurationError
from tkge.data.utils import get_all_days_of_year

import enum
import arrow

from abc import ABC, abstractmethod

SPOT = enum.Enum('spot', ('s', 'p', 'o', 't'))


class DatasetProcessor(ABC, Registrable, Configurable):
    def __init__(self, config: Config):
        Registrable.__init__(self)
        Configurable.__init__(self, config=config)

        self.folder = self.config.get("dataset.folder")
        self.resolution = self.config.get("dataset.temporal.resolution")
        self.index = self.config.get("dataset.temporal.index")
        self.float = self.config.get("dataset.temporal.float")

        self.reciprocal_training = self.config.get("task.reciprocal_training")
        # self.filter_method = self.config.get("data.filter")

        self.train_raw: List[str] = []
        self.valid_raw: List[str] = []
        self.test_raw: List[str] = []

        self.ent2id = defaultdict(None)
        self.rel2id = defaultdict(None)
        self.ts2id = defaultdict(None)

        self.train_set = defaultdict(list)
        self.valid_set = defaultdict(list)
        self.test_set = defaultdict(list)

        self.all_triples = []
        self.all_quadruples = []

        self.data_splits = ["train", "valid", "test"]
        self.data_raw_mappings = {self.data_splits[0]: self.train_raw,
                                  self.data_splits[1]: self.valid_raw,
                                  self.data_splits[2]: self.test_raw}
        self.data_set_mappings = {self.data_splits[0]: self.train_set,
                                  self.data_splits[1]: self.valid_set,
                                  self.data_splits[2]: self.test_set}

        self.load()
        self.process()

    @classmethod
    def create(cls, config: Config):
        """Factory method for data creation"""

        ds_type = config.get("dataset.name")

        if ds_type in DatasetProcessor.list_available():
            kwargs = config.get("dataset.args")  # TODO: 需要改成key的格式
            return DatasetProcessor.by_name(ds_type)(config)
        else:
            raise ConfigurationError(
                f"{ds_type} specified in configuration file is not supported"
                f"implement your data class with `DatasetProcessor.register(name)"
            )

    @abstractmethod
    def process(self):
        raise NotImplementedError

    def index_entities(self, ent: str):
        """
        Associates each given entity with an unique identifier, i.e. the number of different entities.
        """
        if ent not in self.ent2id:
            self.ent2id.update({ent: self.num_entities()})

        return self.ent2id[ent]

    def index_relations(self, rel: str):
        """
        Associates each given relation with an unique identifier, i.e. the number of different relations.
        """
        if rel not in self.rel2id:
            self.rel2id.update({rel: self.num_relations()})

        return self.rel2id[rel]

    def index_timestamps(self, ts):
        """
        Associates each given timestamp with an unique identifier, i.e. the number of different timestamps.
        """
        if ts not in self.ts2id:
            self.ts2id.update({ts: self.num_timestamps()})

        return self.ts2id[ts]

    def load(self):
        """
        Loads the dataset from the train.txt, valid.txt and test.txt from the specified dataset folder.
        Duplicates each quadruple and twists the head and tail entity in the train dataset
        if the flag for reciprocal relations is set.
        """
        train_file = self.folder + "/train.txt"
        valid_file = self.folder + "/valid.txt"
        test_file = self.folder + "/test.txt"

        with open(train_file, "r") as f:
            if self.reciprocal_training:
                for line in f.readlines():
                    self.train_raw.append(line)

                    insert_line = line.split('\t')
                    insert_line[1] += '(RECIPROCAL)'
                    insert_line[0], insert_line[2] = insert_line[2], insert_line[0]
                    insert_line = '\t'.join(insert_line)

                    self.train_raw.append(insert_line)
            else:
                self.train_raw = f.readlines()

            self.train_size = len(self.train_raw)

        with open(valid_file, "r") as f:
            self.valid_raw = f.readlines()

            self.valid_size = len(self.valid_raw)

        with open(test_file, "r") as f:
            self.test_raw = f.readlines()

            self.test_size = len(self.test_raw)

    @abstractmethod
    def process_time(self, origin: str):
        # TODO(gengyuan) use datetime
        raise NotImplementedError

    def get(self, split: str = "train"):
        # TODO(gengyuan)
        return {"train": self.train_set, "valid": self.valid_set, "test": self.test_set}[split]

    def num_entities(self):
        return len(self.ent2id)

    def num_relations(self):
        return len(self.rel2id)

    def num_timestamps(self):
        return len(self.ts2id)

    def filter(self, type="static", target="o") -> Dict[str, List]:
        """
        Returns generated link prediction queries.
        Removes the specified target (either s, p or o) out of a copy of each triple respectively quadruple
        (if specified type is static respectively time-aware) and adds each answer as the last element.
        """
        assert type in ["static",
                        "time-aware",
                        "off"], f"{type} filtering is not implemented; use static/time-aware/off filtering."
        assert target in ["s", "p", "o"], "Only support s(ubject)/p(redicate)/o(bject) prediction task"

        filtered_data = defaultdict(list)

        if type != "off":
            all_tuples = self.all_triples if type == "static" else self.all_quadruples

            for tup in all_tuples:
                query = tup.copy()

                # TODO(gengyuan) enum
                missing = query[SPOT[target].value - 1]
                query[SPOT[target].value - 1] = None

                query_k = f"{query[0]}-{query[1]}-{query[2]}"

                if type == "time-aware":
                    query_k += f"-{query[3]}"

                filtered_data[query_k].append(missing)

        return filtered_data

    def info(self):
        self.config.log(f'==============================================\n'
                        f'Dataset type : {self.config.get("dataset.name")}\n'
                        f"Number of entities : {self.num_entities()}\n"
                        f"Number of relations : {self.num_relations()}\n"
                        f"\n"
                        f"Train set size : {self.train_size}\n"
                        f"Valid set size : {self.valid_size}\n"
                        f"Test set size : {self.test_size}\n"
                        '==============================================')


@DatasetProcessor.register(name="gdelt")
class GDELTDatasetProcessor(DatasetProcessor):
    def __init__(self, config: Config):
        super().__init__(config)

    def process(self):
        """
        Converts the raw text data to meaningful numerical data.
        Since the GDELT dataset already represents the data as numbers (ids), the head and tail entities as well as the
        relations only need to be casted to numerical types.
        """
        for data_split in self.data_splits:
            for rd in self.data_raw_mappings[data_split]:
                head, rel, tail, ts = rd.strip().split('\t')
                head = int(head)
                rel = int(rel)
                tail = int(tail)
                ts = self.process_time(ts)
                ts_id = self.index_timestamps(ts)

                self.data_set_mappings[data_split]['triple'].append([head, rel, tail])
                self.data_set_mappings[data_split]['timestamp_id'].append([ts_id])
                self.data_set_mappings[data_split]['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

                self.all_triples.append([head, rel, tail])
                self.all_quadruples.append([head, rel, tail, ts_id])

    def process_time(self, origin: str, resolution: str = 'day'):
        all_resolutions = ['year', 'month', 'day', 'hour', 'minute', 'second']
        assert resolution in all_resolutions, f"Time granularity should be {all_resolutions}"

        ts = origin.split('-') + ['00', '00', '00']
        ts = ts[:all_resolutions.index(resolution) + 1]
        ts = '-'.join(ts)

        return ts


@DatasetProcessor.register(name="icews14")
class ICEWS14DatasetProcessor(DatasetProcessor):
    def process(self):
        """
        Converts the raw text data to meaningful numerical data.
        Since the ICEWS14 dataset represent its data in raw semantic texts, the head and tail entities as well as the
        relations need to be indexed programmatically.
        """
        all_timestamp = get_all_days_of_year(2014)
        self.ts2id = {ts: (arrow.get(ts) - arrow.get('2014-01-01')).days for ts in all_timestamp}

        for data_split in self.data_splits:
            for rd in self.data_raw_mappings[data_split]:
                head, rel, tail, ts = rd.strip().split('\t')
                head = self.index_entities(head)
                rel = self.index_relations(rel)
                tail = self.index_entities(tail)
                ts = self.process_time(ts)
                ts_id = self.index_timestamps(ts)

                self.data_set_mappings[data_split]['triple'].append([head, rel, tail])
                self.data_set_mappings[data_split]['timestamp_id'].append([ts_id])
                self.data_set_mappings[data_split]['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

                self.all_triples.append([head, rel, tail])
                self.all_quadruples.append([head, rel, tail, ts_id])

    def process_time(self, origin: str):
        all_resolutions = ['year', 'month', 'day', 'hour', 'minute', 'second']
        assert self.resolution in all_resolutions, f"Time granularity should be {all_resolutions}"

        ts = origin.split('-') + ['00', '00', '00']
        ts = ts[:all_resolutions.index(self.resolution) + 1]
        ts = '-'.join(ts)

        return ts


@DatasetProcessor.register(name="icews05-15")
class ICEWS0515DatasetProcessor(DatasetProcessor):
    def process(self):
        """
        Converts the raw text data to meaningful numerical data.
        Since the ICEWS05-15 dataset represent its data in raw semantic texts, the head and tail entities as well as the
        relations need to be indexed programmatically.
        """
        for data_split in self.data_splits:
            for rd in self.data_raw_mappings[data_split]:
                head, rel, tail, ts = rd.strip().split('\t')
                head = self.index_entities(head)
                rel = self.index_relations(rel)
                tail = self.index_entities(tail)
                ts = self.process_time(ts)

                self.data_set_mappings[data_split]['triple'].append([head, rel, tail])
                self.data_set_mappings[data_split]['timestamp_id'].append([self.index_timestamps(ts)])
                self.data_set_mappings[data_split]['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

    def process_time(self, origin: str):
        raise NotImplementedError


@DatasetProcessor.register(name="yago15k")
class YAGO15KDatasetProcessor(DatasetProcessor):
    def process(self):
        """
        Processes the raw data for each data type (i.e. train, valid and test) of the YAGO15k dataset.
        If a fact has a temporal part (temporal modifier and timestamp), then the temporal modifier is used to
        concatenate the relation, so a relation of a fact can be '<relation>occursSince', '<relation>occursUntil' or
        '<relation>_no-time'.
        """
        for data_split in self.data_splits:
            for rd in self.data_raw_mappings[data_split]:
                fact = rd.strip().split('\t')
                # self.config.log(f"Processing fact in line {index + 1}: {fact}")

                if len(fact) > 4:
                    head, rel, tail, mod, ts = fact
                    rel += mod
                    ts = ts.split('-')[0][1:]
                    ts_id = self.index_timestamps(ts)
                elif len(fact) == 4:
                    # ignore the two tuples with temporal modifiers but without timestamp
                    continue
                else:
                    head, rel, tail = fact
                    rel += '_no-time'
                    ts = 'no-time'
                    ts_id = self.index_timestamps(ts)

                head_id = self.index_entities(head)
                rel_id = self.index_relations(rel)
                tail_id = self.index_entities(tail)

                self.data_set_mappings[data_split]['triple'].append([head_id, rel_id, tail_id])
                self.data_set_mappings[data_split]['timestamp_id'].append([ts_id])
                self.data_set_mappings[data_split]['timestamp_float'].append([int(ts) if ts != 'no-time' else 0])

                self.all_triples.append([head_id, rel_id, tail_id])
                self.all_quadruples.append([head_id, rel_id, tail_id, ts_id])

    def process_time(self, origin: str):
        raise NotImplementedError


class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dict[str, List], datatype: Optional[List[str]] = None):
        super().__init__()

        self.dataset = dataset
        self.datatype = datatype

        # TODO(gengyuan) assert the lengths of all lists in self.dataset
        # assert all( for i in dataset.items())

    def __len__(self):
        # TODO(gengyuan) calculate the length
        return len(self.dataset['triple'])

    def __getitem__(self, index, train=True):
        sample = torch.Tensor(self.dataset['triple'][index])

        for type in self.datatype:
            if type == 'timestamp_id':
                timestamp_id = torch.Tensor(self.dataset['timestamp_id'][index])
                sample = torch.cat([sample, timestamp_id], dim=0)

            elif type == 'timestamp_float':
                timestamp_float = torch.Tensor(self.dataset['timestamp_float'][index])
                sample = torch.cat([sample, timestamp_float], dim=0)
            else:
                raise NotImplementedError

        return sample
