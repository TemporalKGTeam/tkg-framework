import torch
from torch.utils.data.dataset import Dataset as PTDataset

from typing import Dict, List, Tuple, Optional
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
        self.config.log('==============================================')
        self.config.log(f'Dataset type : {self.config.get("dataset.name")}')
        self.config.log(f"Number of entities : {self.num_entities()}")
        self.config.log(f"Number of relations : {self.num_relations()}")
        self.config.log(f"\n")
        self.config.log(f"Train set size : {self.train_size}")
        self.config.log(f"Valid set size : {self.valid_size}")
        self.config.log(f"Test set size : {self.test_size}")
        self.config.log('==============================================')


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
        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = int(head)
            rel = int(rel)
            tail = int(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts_id])
            self.train_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.valid_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = int(head)
            rel = int(rel)
            tail = int(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([ts_id])
            self.valid_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.test_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = int(head)
            rel = int(rel)
            tail = int(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([ts_id])
            self.test_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

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

        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts_id])
            self.train_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.valid_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([ts_id])
            self.valid_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.test_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([ts_id])
            self.test_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

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
        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([self.index_timestamps(ts)])
            self.train_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([self.index_timestamps(ts)])
            self.valid_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([self.index_timestamps(ts)])
            self.test_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

    def process_time(self, origin: str):
        raise NotImplementedError


#
# @DatasetProcessor.register(name="wiki")
# class WIKIDatasetProcessor(DatasetProcessor):
#     def process(self):
#         pass
#
#     def process_time(self, origin: str):
#         pass
#
#
# @DatasetProcessor.register(name="wiki12k")
# class WIKI12KDatasetProcessor(DatasetProcessor):
#     def process(self):
#         pass
#
#     def process_time(self, origin: str):
#         pass
#
#
# @DatasetProcessor.register(name="yago")
# class YAGODatasetProcessor(DatasetProcessor):
#     def process(self):
#         pass
#
#     def process_time(self, origin: str):
#         pass
#
#
# @DatasetProcessor.register(name="yago11k")
# class YAGO11KDatasetProcessor(DatasetProcessor):
#     def process(self):
#         pass
#
#     def process_time(self, origin: str):
#         pass


@DatasetProcessor.register(name="yago15k")
class YAGO15KDatasetProcessor(DatasetProcessor):
    def process(self):
        """
        Processes the raw data for each data type (i.e. train, valid and test) of the YAGO15k dataset.
        If a fact has a temporal part (temporal modifier and timestamp), then the temporal modifier is used to
        concatenate the relation, so a relation of a fact can be '<relation>occursSince', '<relation>occursUntil' or
        '<relation>_no-time'.
        """
        data_splits = ["train", "valid", "test"]
        data_raw_mappings = {data_splits[0]: self.train_raw, data_splits[1]: self.valid_raw, data_splits[2]: self.test_raw}
        data_set_mappings = {data_splits[0]: self.train_set, data_splits[1]: self.valid_set, data_splits[2]: self.test_set}

        for data_split in data_splits:
            for rd in data_raw_mappings[data_split]:
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

                data_set_mappings[data_split]['triple'].append([head_id, rel_id, tail_id])
                data_set_mappings[data_split]['timestamp_id'].append([ts_id])
                data_set_mappings[data_split]['timestamp_float'].append([int(ts) if ts != 'no-time' else 0])

                self.all_triples.append([head_id, rel_id, tail_id])
                self.all_quadruples.append([head_id, rel_id, tail_id, ts_id])

                # year_start, year_stop, index = self.process_time(data_split, index=index, fact=fact, triple=triple)
                #
                # for y in range(year_start, year_stop + 1):
                #     ts_id = self.index_timestamps(y)
                #
                #     data_set_mappings[data_split]['triple'].append([head_id, rel_id, tail_id])
                #     data_set_mappings[data_split]['timestamp_id'].append([ts_id])
                #     data_set_mappings[data_split]['timestamp_float'].append([y])
                #
                #     self.all_triples.append([head_id, rel_id, tail_id])
                #     self.all_quadruples.append([head_id, rel_id, tail_id, ts_id])
                #
                #     # self.config.log(f" {data_split} tuple: {str([head_id, rel_id, tail_id, ts_id, [y]])}")

    def process_time(self, origin: str):
        raise NotImplementedError

    # def process_time(self, origin: str, index: int = 0, fact: List[str] = None, triple: Tuple = None):
    #     """
    #     Processes the time information in the YAGO15k dataset.
    #     Since there are not only timestamps but also time modifiers, namely 'occursSince' and 'occursUntil', and
    #     corresponding tuples that appear twice in a row, a few cases need to be addressed:
    #     - Case 1: The timespan is given by the data.
    #     - Case 2: The timespan is not given explicitly, but with argument-less modifiers that indicate universal truth.
    #     - Case 3: The timespan is partly given by the data, i.e. the start time is known.
    #     - Case 4: The timespan is partly given by the data, i.e. the end time is known.
    #     - Case 5: There is no time information given at all.
    #     In cases 1 and 2 the index needs to be incremented by 1 because the next line was already processed then.
    #     Returns the start and end timestamp (only as the year) of the triple, as well as the maybe modified index.
    #     """
    #     data = {"train": self.train_raw, "valid": self.valid_raw, "test": self.test_raw}
    #     data_raw = data[origin]
    #     valid_temp_mods = ["occursSince", "occursUntil"]
    #
    #     # process time of the current fact
    #     temp_mod = fact[3][1:-1] if len(fact) >= 4 and fact[3][1:-1] in valid_temp_mods else None
    #     year = int(fact[4].split('-')[0][1:]) if len(fact) == 5 else 0
    #
    #     # process time of the next fact only if there exists at least one more fact after this
    #     if index + 1 < len(data_raw):
    #         next_fact = data_raw[index + 1].strip().split('\t')
    #         next_triple = next_fact[0][1:-1], next_fact[1][1:-1], next_fact[2][1:-1]
    #
    #         next_temp_mod = next_fact[3][1:-1] if len(next_fact) >= 4 and next_fact[3][1:-1] in valid_temp_mods else None
    #         next_year = int(next_fact[4][1:].split('-')[0]) if len(next_fact) == 5 else 0
    #
    #         is_closed_timespan = triple == next_triple and temp_mod and year != 0 and next_temp_mod and next_year != 0
    #     else:
    #         next_temp_mod = None
    #         next_year = 0
    #
    #         is_closed_timespan = False
    #
    #     if is_closed_timespan:
    #         # case 1: same triple appears twice in a row and has a temporal modifier as well as a timestamp
    #         year_stop = next_year
    #         index += 1
    #     elif temp_mod and next_temp_mod and year == 0 and next_year == 0:
    #         # case 2: same triple appears twice in a row but has only a temporal modifier and no timestamp
    #         year = int(self.config.get('dataset.args.year_min'))
    #         year_stop = int(self.config.get('dataset.args.year_max'))
    #         index += 1
    #     elif temp_mod and temp_mod == 'occursSince':
    #         # case 3: triple appears only once and is true until now (or forever from the start year)
    #         year_stop = int(self.config.get('dataset.args.year_max'))
    #     elif temp_mod and temp_mod == 'occursUntil':
    #         # case 4: triple appears only once and was true until a certain year
    #         year_stop = year
    #         year = int(self.config.get('dataset.args.year_min'))
    #     else:
    #         # case 5: otherwise there is no timespan, i.e. a stand alone triple
    #         year_stop = year
    #
    #     return year, year_stop, index


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
