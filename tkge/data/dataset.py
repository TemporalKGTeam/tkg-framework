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
import datetime
import calendar

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

        self.train_size = len(self.train_raw)
        self.valid_size = len(self.valid_raw)
        self.test_size = len(self.test_raw)

        self.data_splits = ["train", "valid", "test"]
        self.data_raw_mappings = {self.data_splits[0]: self.train_raw,
                                  self.data_splits[1]: self.valid_raw,
                                  self.data_splits[2]: self.test_raw}
        self.data_set_mappings = {self.data_splits[0]: self.train_set,
                                  self.data_splits[1]: self.valid_set,
                                  self.data_splits[2]: self.test_set}

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

    def index_quadruple(self, quadruple: List[str]):
        head_id, rel_id, tail_id = self.index_triple(quadruple[0:3])
        ts_id = self.index_timestamps(quadruple[3])

        return head_id, rel_id, tail_id, ts_id

    def index_triple(self, triple: List[str]):
        head_id = self.index_entities(triple[0])
        rel_id = self.index_relations(triple[1])
        tail_id = self.index_entities(triple[2])

        return head_id, rel_id, tail_id

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

        with open(valid_file, "r") as f:
            self.valid_raw = f.readlines()

        with open(test_file, "r") as f:
            self.test_raw = f.readlines()

    @abstractmethod
    def process_time(self, origin: str):
        # TODO(gengyuan) use datetime
        raise NotImplementedError

    def get(self, split: str = "train"):
        # TODO(gengyuan)
        return {"train": self.train_set, "valid": self.valid_set, "test": self.test_set}[split]

    def add(self, data_split, head_id, rel_id, tail_id, ts_id, ts_float):
        self.data_set_mappings[data_split]['triple'].append([head_id, rel_id, tail_id])
        self.data_set_mappings[data_split]['timestamp_id'].append([ts_id])
        self.data_set_mappings[data_split]['timestamp_float'].append(ts_float)

        self.all_triples.append([head_id, rel_id, tail_id])
        self.all_quadruples.append([head_id, rel_id, tail_id, ts_id])

    def num_entities(self):
        return len(self.ent2id)

    def num_relations(self):
        return len(self.rel2id)

    def num_timestamps(self):
        return len(self.ts2id)

    def num_time_identifier(self):
        return len(self.ts2id)

    def filter(self, type="static", target="o") -> Dict[str, List]:
        """
        Returns generated link prediction queries.
        Removes the specified target (either s, p or o) out of a copy of each triple respectively quadruple
        (if specified type is static respectively time-aware) and adds each answer as the last element.
        """
        self.config.assert_true(type in ["static",
                        "time-aware",
                        "off"], f"{type} filtering is not implemented; use static/time-aware/off filtering.")
        self.config.assert_true(target in ["s", "p", "o"], "Only support s(ubject)/p(redicate)/o(bject) prediction task")

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
                        f"Number of temporal identifiers : {self.num_timestamps()}"
                        f"\n"
                        f"Train set size : {self.train_size}\n"
                        f"Valid set size : {self.valid_size}\n"
                        f"Test set size : {self.test_size}\n"
                        f'==============================================')


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
                ts = self.process_time(ts)

                head_id = int(head)
                rel_id = int(rel)
                tail_id = int(tail)
                ts_id = self.index_timestamps(ts)
                ts_float = list(map(lambda x: int(x), ts.split('-')))

                self.add(data_split, head_id, rel_id, tail_id, ts_id, ts_float)

    def process_time(self, origin: str, resolution: str = 'day'):
        all_resolutions = ['year', 'month', 'day', 'hour', 'minute', 'second']
        self.config.assert_true(resolution in all_resolutions, f"Time granularity should be {all_resolutions}")

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
                ts = self.process_time(ts)

                head_id, rel_id, tail_id, ts_id = self.index_quadruple([head, rel, tail, ts])
                ts_float = list(map(lambda x: int(x), ts.split('-')))

                self.add(data_split, head_id, rel_id, tail_id, ts_id, ts_float)

    def process_time(self, origin: str):
        all_resolutions = ['year', 'month', 'day', 'hour', 'minute', 'second']
        self.config.assert_true(self.resolution in all_resolutions, f"Time granularity should be {all_resolutions}")

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
                ts = self.process_time(ts)

                head_id, rel_id, tail_id, ts_id = self.index_quadruple([head, rel, tail, ts])
                ts_float = list(map(lambda x: int(x), ts.split('-')))

                self.add(data_split, head_id, rel_id, tail_id, ts_id, ts_float)

    def process_time(self, origin: str):
        all_resolutions = ['year', 'month', 'day', 'hour', 'minute', 'second']
        self.config.assert_true(self.resolution in all_resolutions, f"Time granularity should be {all_resolutions}")

        ts = origin.split('-') + ['00', '00', '00']
        ts = ts[:all_resolutions.index(self.resolution) + 1]
        ts = '-'.join(ts)

        return ts


# TODO test it with ATisE and HyTE when they work properly
@DatasetProcessor.register(name="yago11k")
class YAGO11KDatasetProcessor(DatasetProcessor):
    def process(self):
        """
        Processes the raw data for each data type (i.e. train, valid and test) of the YAGO11k dataset.
        Every fact is framed into its time interval through adding facts for each timestamp between the start timestamp
        and the end timestamp. Use the year_max and year_min parameters to set the year if the start timestamp or
        end timestamp have no time information (i.e. ####-##-##).
        The specified granularity will be augmented to the first possible day and/or month respectively the last
        possible day and/or month for timestamps of the form XXXX-##-## and XXXX-XX-##.
        """
        if self.resolution == "day":
            self.config.log(f"Processing facts with day resolution on dataset yago11k will consume a lot of time.", "warning")

        for data_split in self.data_splits:
            i = 0
            for rd in self.data_raw_mappings[data_split]:
                i += 1
                fact = rd.strip().split('\t')

                head_id, rel_id, tail_id = self.index_triple(fact[:3])

                time_interval = self.process_time(origin='', start_ts=fact[3], end_ts=fact[4])

                for ts in time_interval:
                    ts_id = self.index_timestamps(ts)
                    ts_float = list(map(int, ts.split('-'))) if not isinstance(ts, int) else [ts]
                    self.add(data_split, head_id, rel_id, tail_id, ts_id, ts_float)

    def process_time(self, origin: str, start_ts=None, end_ts=None):
        all_resolutions = ['year', 'month', 'day']
        self.config.assert_true(self.resolution in all_resolutions, f"Time granularity should be {all_resolutions}"
                                                                    f" (smaller resolutions consume too much time for"
                                                                    f"processing and a lot of memory"
                                                                    f" regarding yago11k).")

        # process the different timestamp representations to logical units
        start = self.__timestamp_as_list(start_ts)
        end = self.__timestamp_as_list(end_ts)

        # complete incomplete start timestamps
        if len(start) == 0:
            start = [self.config.get('dataset.args.year_min')]
        if len(start) == 1:
            # if only year is available, start at first possible day of that year
            start.extend([1, 1])
        if len(start) == 2:
            # if year and month is available, start at first day of that month
            start.append(1)

        # complete incomplete end timestamps
        if len(end) == 0:
            end = [self.config.get('dataset.args.year_max')]
        if len(end) == 1:
            # if only year is available, stop at last possible point
            end.extend([12, 31])
        if len(end) == 2:
            # if year and month is available, stop at last point of that month
            end.append(calendar.monthrange(int(end[0]), int(end[1]))[1])

        # there's at least one month greater than 12 (13047)
        end[1] = end[1] % 12 + 1 if end[1] > 12 else end[1]

        if self.resolution == "year":
            return list(range(start[0], end[0] + 1))

        if self.resolution == "month":
            all_ts = []
            for year in range(start[0], end[0] + 1):
                start_month = 1
                end_month = 12

                if year == start[0]:
                    start_month = start[1]
                if year == end[0]:
                    end_month = end[1]

                for month in range(start_month, end_month + 1):
                    all_ts.append(f"{year}-{month}")

            return all_ts

        # for day resolution, calculate it via the datetime library (costs a lot of time)
        start_date = datetime.datetime(*start)
        end_date = datetime.datetime(*end)
        delta = datetime.timedelta(days=1)

        all_ts = []
        while start_date <= end_date:
            all_ts.append(start_date.strftime('%Y-%m-%d'))
            start_date += delta

        return all_ts

    @staticmethod
    def __timestamp_as_list(ts):
        if ts[0] == '-':
            ts_list = list(filter(lambda x: x != '##' and x != '####', ts.split('-')[1:]))
            ts_list = [x.replace('#', '0') for x in ts_list]
            ts_list = list(map(int, ts_list))
            ts_list[0] = 1
        else:
            ts_list = list(filter(lambda x: x != '##' and x != '####', ts.split('-')))
            ts_list = [x.replace('#', '0') for x in ts_list]
            ts_list = list(map(int, ts_list))

        return ts_list


@DatasetProcessor.register(name="yago15k")
class YAGO15KDatasetProcessor(DatasetProcessor):
    def process(self):
        """
        Processes the raw data for each data type (i.e. train, valid and test) of the YAGO15k dataset.
        If a fact has a temporal part (temporal modifier and timestamp), then the temporal modifier is concatenated to
        the relation, so a relation of a fact can be '<relation>occursSince', '<relation>occursUntil' or
        '<relation>_no-time'.
        """
        self.config.assert_true(self.resolution == "year", f"Time granularity should be year for yago15k.")

        for data_split in self.data_splits:
            for rd in self.data_raw_mappings[data_split]:
                fact = rd.strip().split('\t')
                # self.config.log(f"Processing fact in line {index + 1}: {fact}")

                if len(fact) > 4:
                    head, rel, tail, mod, ts = fact
                    rel += mod
                    ts = ts.split('-')[0][1:]
                    ts_float = int(ts)
                elif len(fact) == 4:
                    # ignore the two tuples with temporal modifiers but without timestamp
                    continue
                else:
                    head, rel, tail = fact
                    rel += '_no-time'
                    ts = 'no-time'
                    ts_float = 0

                head_id, rel_id, tail_id, ts_id = self.index_quadruple([head, rel, tail, ts])

                self.add(data_split, head_id, rel_id, tail_id, ts_id, [ts_float])

    def process_time(self, origin: str):
        raise NotImplementedError


@DatasetProcessor.register(name="wikidata_lse")
class WIKIDATALSEDatasetProcessor(DatasetProcessor):
    def process(self):
        """
        Processes the raw data for each data type (i.e. train, valid and test) of the Wikidata dataset that was used
        in the "Learning Sequence Encoders for Temporal Knowledge Graph Completion" paper,
        see https://arxiv.org/abs/1809.03202.
        The temporal modifier is concatenated to the relation.
        """
        for data_split in self.data_splits:
            for rd in self.data_raw_mappings[data_split]:
                fact = rd.strip().split('\t')

                head, rel, tail, mod, ts = fact
                rel += mod

                head_id, rel_id, tail_id, ts_id = self.index_quadruple([head, rel, tail, ts])
                ts_float = int(ts)

                self.add(data_split, head_id, rel_id, tail_id, ts_id, [ts_float])

    def process_time(self, origin: str):
        raise NotImplementedError


class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dict[str, List], datatype: Optional[List[str]] = None):
        super().__init__()

        self.dataset = dataset
        self.datatype = datatype

        # TODO(gengyuan) assert the lengths of all lists in self.dataset
        # use self.config.assert_true(condition, message)
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
