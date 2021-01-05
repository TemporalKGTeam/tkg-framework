import torch
from torch.utils.data.dataset import Dataset as PTDataset
import numpy as np

from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from tkge.common.registry import Registrable
from tkge.common.config import Config
from tkge.common.error import ConfigurationError

import enum

SPOT = enum.Enum('spot', ('s', 'p', 'o', 't'))


class DatasetProcessor(Registrable):
    def __init__(self, config: Config):
        super().__init__(config)

        self.folder = self.config.get("dataset.folder")
        self.level = self.config.get("dataset.temporal.level")
        self.index = self.config.get("dataset.temporal.index")
        self.float = self.config.get("dataset.temporal.float")
        # self.filter_method = self.config.get("data.filter")

        self.train_raw = None
        self.valid_raw = None
        self.test_raw = None

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

    @staticmethod
    def create(config: Config):
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

    def process(self):
        raise NotImplementedError

    def index_entities(self, ent):
        if ent not in self.ent2id:
            self.ent2id.update({ent: self.num_entities()})

        return self.ent2id[ent]

    def index_relations(self, rel):
        if rel not in self.rel2id:
            self.rel2id.update({rel: self.num_relations()})

        return self.rel2id[rel]

    def index_timestamps(self, ts):
        if ts not in self.ts2id:
            self.ts2id.update({ts: self.num_timestamps()})

        return self.ts2id[ts]

    def load(self):
        train_file = self.folder + "/train.txt"
        valid_file = self.folder + "/valid.txt"
        test_file = self.folder + "/test.txt"

        with open(train_file, "r") as f:
            self.train_raw = f.readlines()

        with open(valid_file, "r") as f:
            self.valid_raw = f.readlines()

        with open(test_file, "r") as f:
            self.test_raw = f.readlines()

    def process_time(self, origin: str):
        # TODO(gengyuan) 使用time或者datetime里面的函数来处理
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


@DatasetProcessor.register(name="gdelt")
class GDELTDatasetProcessor(DatasetProcessor):
    def process(self):
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

    def process_time(self, origin: str, granularity: str = 'day'):
        level = ['year', 'month', 'day', 'hour', 'minute', 'second']
        assert granularity in level, f"Time granularity should be {level}"

        ts = origin.split('-') + ['00', '00', '00']
        ts = ts[:level.index(granularity) + 1]
        ts = '-'.join(ts)

        return ts


@DatasetProcessor.register(name="icews14")
class ICEWS14DatasetProcessor(DatasetProcessor):
    def process(self):
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
        level = ['year', 'month', 'day', 'hour', 'minute', 'second']
        assert self.level in level, f"Time granularity should be {level}"

        ts = origin.split('-') + ['00', '00', '00']
        ts = ts[:level.index(self.level) + 1]
        ts = '-'.join(ts)

        return ts


@DatasetProcessor.register(name="icews05-15")
class ICEWS0515DatasetProcessor(DatasetProcessor):
    def process(self):
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
        level = ['year', 'month', 'day', 'hour', 'minute', 'second']
        assert self.level in level, f"Time granularity should be {level}"

        ts = origin.split('-') + ['00', '00', '00']
        ts = ts[:level.index(self.level) + 1]
        ts = '-'.join(ts)

        return ts


@DatasetProcessor.register(name="wiki")
class WIKIDatasetProcessor(DatasetProcessor):
    def process(self):
        pass

    def process_time(self, origin: str):
        pass


@DatasetProcessor.register(name="yago")
class YAGODatasetProcessor(DatasetProcessor):
    def process(self):
        pass

    def process_time(self, origin: str):
        pass


# class Dataset(Registrable):
#     def __init__(self, config: Config, require_tsid: bool = True):
#         super().__init__(config)
#
#         self.require_tsid = require_tsid  # TODO 需要吗
#
#         self.folder = config.get("data.folder")
#
#         self.ent2id = defaultdict(dict)
#         self.rel2id = defaultdict(dict)
#         self.ts2id = defaultdict(dict)
#
#         self.train_quadruples = []
#         self.valid_quadruples = []
#         self.test_quadruples = []
#
#         self.train_timestamps = []
#         self.valid_timestamps = []
#         self.test_timestamps = []
#
#         self._prepare_dataset(config)
#
#     @staticmethod
#     def create(config: Config):
#         """Factory method for data creation"""
#
#         ds_type = config.get("data.name")
#
#         if ds_type in Dataset.list_available():
#             kwargs = config.get("data.args")  # TODO: 需要改成key的格式
#             return Dataset.by_name(ds_type)(config)
#         else:
#             raise ConfigurationError(
#                 f"{ds_type} specified in configuration file is not supported"
#                 f"implement your data class with `Dataset.register(name)"
#             )
#
#     def _prepare_dataset(self, config: Config):
#
#         if_mapping = config.get("data.mapping")
#
#         # TODO do mapping
#         # if if_mapping:
#         #     raise NotImplementedError
#         #     # self._load_map()
#         # else:
#         #     self._create_map()
#
#         self._load_data()
#
#     def _load_data(self):
#         train_file = self.folder + "/train.txt"
#         valid_file = self.folder + "/valid.txt"
#         test_file = self.folder + "/test.txt"
#
#         with open(train_file, "r") as f:
#             train_data = f.readlines()
#
#         with open(valid_file, "r") as f:
#             valid_data = f.readlines()
#
#         with open(test_file, "r") as f:
#             test_data = f.readlines()
#
#         train_id_tuple = []
#         valid_id_tuple = []
#         test_id_tuple = []
#
#         train_str_ts = []
#         valid_str_ts = []
#         test_str_ts = []
#
#         for data in train_data:
#             h, r, t, ts = data.strip().split("\t")
#
#             ts = self.process_timestamp(ts)
#             y, m, d = list(map(lambda x: int(x), ts.split('-')))
#
#             quatruple = [self.index_entities(h), self.index_relations(r), self.index_entities(t)]
#
#             # self.index_timestamps(ts)]
#
#             train_id_tuple.append(quatruple)
#             train_str_ts.append([y, m, d])
#
#         for data in valid_data:
#             h, r, t, ts = data.strip().split("\t")
#
#             quatruple = [self.index_entities(h), self.index_relations(r), self.index_entities(t)]
#
#             # self.index_timestamps(ts)]
#
#             valid_id_tuple.append(quatruple)
#             valid_str_ts.append([y, m, d])
#
#         for data in test_data:
#             h, r, t, ts = data.strip().split("\t")
#
#             quatruple = [self.index_entities(h), self.index_relations(r), self.index_entities(t)]
#
#             # self.index_timestamps(ts)]
#
#             test_id_tuple.append(quatruple)
#             test_str_ts.append([y, m, d])
#
#         self.train_quadruples = train_id_tuple
#         self.valid_quadruples = valid_id_tuple
#         self.test_quadruples = test_id_tuple
#
#         self.train_timestamps = train_str_ts
#         self.valid_timestamps = valid_str_ts
#         self.test_timestamps = test_str_ts
#
#     def _get_train(self):
#         return np.array(self.train_quadruples), np.array(self.train_timestamps)
#
#     def get_train(self):
#         return SplitDataset(self._get_train())
#
#     def _get_valid(self):
#         return np.array(self.valid_quadruples), np.array(self.valid_timestamps)
#
#     def get_valid(self):
#         return SplitDataset(self._get_valid())
#
#     def _get_test(self):
#         return np.array(self.test_quadruples), np.array(self.test_timestamps)
#
#     def get_test(self):
#         return SplitDataset(self._get_test())
#
#     def process_timestamp(self, ts, granularity: str = "yyyy-mm-dd"):
#         form_list = ["yyyy-mm-dd", "yyyy-mm", "yyyy", "yyyy-span"]
#
#         if granularity not in form_list:
#             raise ConfigurationError(f"timestamp granularity should be in form {form_list}")
#
#         # TODO re
#
#         return ts
#
#     def index_entities(self, ent):
#         if ent not in self.ent2id:
#             self.ent2id.update({ent: self.num_entities()})
#
#         return self.ent2id[ent]
#
#     def index_relations(self, rel):
#         if rel not in self.rel2id:
#             self.rel2id.update({rel: self.num_relations()})
#
#         return self.rel2id[rel]
#
#     def index_timestamps(self, ts):
#         if ts not in self.ts2id:
#             self.ts2id.update({ts: self.num_timestamps()})
#
#         return self.ts2id[ts]
#
#     def _load_map(self):
#         raise NotImplementedError
#
#     def num_entities(self):
#         return len(self.ent2id)
#
#     def num_relations(self):
#         return len(self.rel2id)
#
#     def num_timestamps(self):
#         return len(self.ts2id)


# @Dataset.register(name="icews14")
# class Icews14Dataset(Dataset):
#     def __init__(self, config: Config):
#         super().__init__(config=config)
#
#
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

# class Dataset(Configurable):
#     def __init__(self, config: Config, folder: str):
#         super().__init__(self, config, configuration_key="data")
#
#         self.folder = folder
#
#         try:
#             self._num_entities: int = config.get("data.num_entities")
#             if self._num_entities < 0:
#                 self._num_entities = None
#         except ConfigurationError:
#             self._num_entities: int = None
#
#         try:
#             self._num_relations: int = config.get("data.num_relations")
#             if self._num_relations < 0:
#                 self._num_relations = None
#         except ConfigurationError:
#             self._num_relations: int = None
#
#         try:
#             self._num_timestamps: int = config.get("data.num_timestamps")
#             if self._num_timestamps < 0:
#                 self._num_timestamps = None
#         except ConfigurationError:
#             self._num_timestamps: int = None
#
#         #: split-name to (n,3) int32 tensor
#         self._triples: Dict[str, Tensor] = {}
#         # self._quadruples: Dict[str, Tensor] = {}
#
#         #: meta data that is part if this data. Indexed by key.
#         self._meta: Dict[str, Any] = {}
#
#         #: data derived automatically from the splits or meta data. Indexed by key.
#         self._indexes: Dict[str, Any] = {}
#
#         #: functions that compute and add indexes as needed; arguments are data and
#         # key. : Indexed by key (same key as in self._indexes)
#         self.index_functions: Dict[str, Callable] = {}
#         create_default_index_functions(self)
#
#     # Load
#     @staticmethod
#     def load(config: Config, preload_data=True):
#         """Loads a data.
#
#         If preload_data is set, loads entity and relation maps as well as all splits.
#         Otherwise, this data is lazy loaded on first use.
#
#         """
#         name = config.get("data.name")
#         folder = os.path.join(tkge_base_dir(), "data", name)
#         if os.path.isfile(os.path.join(folder, "data.yaml")):
#             config.log("Loading configuration of data " + name + "...")
#             config.load(os.path.join(folder, "data.yaml"))
#
#         data = Dataset(config, folder)
#         if preload_data:
#             data.entity_ids()
#             data.relation_ids()
#             data.timestamp_ids()
#             for split in ["train", "valid", "test"]:
#                 data.split(split)
#         return data
#
#     @staticmethod
#     def _load_triples(filename: str, delimiter="\t") -> Tensor:
#         triples = np.loadtxt(filename, usecols=range(0, 3), dtype=int)
#         return torch.from_numpy(triples)
#
#     @staticmethod
#     def _load_quadruples(filename: str, delimiter="\t") -> Tensor:
#         quadruples = np.loadtxt(filename, usecols=range(0, 4), dtype=int)
#         return torch.from_numpy(quadruples)
#
#     def get(self, key: str) -> Tensor:
#         "Load or return the triples or quadruples with the specified key."
#         if key not in self._triples:
#             filename = self.config.get(f"data.files.{key}.filename")
#             filetype = self.config.get(f"data.files.{key}.type")
#             if filetype == "triples":
#                 triples = Dataset._load_triples(os.path.join(self.folder, filename))
#
#             elif filetype == "quadruples":
#                 triples = Dataset._load_quadruples(os.path.join(self.folder, filename))
#             else:
#                 raise ValueError(
#                     "Unexpected file type: "
#                     f"data.files.{key}.type='{filetype}', expected 'triples'"
#                 )
#             self.config.log(f"Loaded {len(triples)} {key} triples")
#             self._triples[key] = triples
#
#         return self._triples[key]
#
#     @staticmethod
#     def _load_map(
#             filename: str,
#             as_list: bool = False,
#             delimiter: str = "\t",
#             ignore_duplicates=False,
#     ) -> Union[List, Dict]:
#         n = 0
#         dictionary = {}
#         warned_overrides = False
#         duplicates = 0
#         with open(filename, "r", encoding="utf-8") as file:
#             for line in file:
#                 key, value = line.split(delimiter, maxsplit=1)
#                 value = value.rstrip("\n")
#                 if as_list:
#                     key = int(key)
#                     n = max(n, key + 1)
#                 if key in dictionary:
#                     duplicates += 1
#                     if not ignore_duplicates:
#                         raise KeyError(f"{filename} contains duplicated keys")
#                 else:
#                     dictionary[key] = value
#         if as_list:
#             array = [None] * n
#             for index, value in dictionary.items():
#                 array[index] = value
#             return array, duplicates
#         else:
#             return dictionary, duplicates
#
#     def load_map(
#             self,
#             key: str,
#             as_list: bool = False,
#             maptype=None,
#             ids_key=None,
#             ignore_duplicates=False,
#     ) -> Union[List, Dict]:
#         """Load or return the map with the specified key.
#
#         If `as_list` is set, the map is converted to an array indexed by the map's keys.
#
#         If `maptype` is set ensures that the map being loaded has the specified type.
#         Valid map types are `map` (keys are indexes) and `idmap` (keys are ids).
#
#         If the map is of type `idmap`, its keys can be converted to indexes by setting
#         `ids_key` to either `entity_ids` or `relation_ids` and `as_list` to `True`.
#
#         If ignore_duplicates is set to `False` and the map contains duplicate keys,
#         raise a `KeyError`. Otherwise, logs a warning and picks first occurrence of a
#         key.
#
#         """
#         if key not in self._meta:
#             filename = self.config.get(f"data.files.{key}.filename")
#             filetype = self.config.get(f"data.files.{key}.type")
#             if (maptype and filetype != maptype) or (
#                     not maptype and filetype not in ["map", "idmap"]
#             ):
#                 if not maptype:
#                     maptype = "map' or 'idmap"
#                 raise ConfigurationError(
#                     "Unexpected file type: "
#                     f"data.files.{key}.type='{filetype}', expected {maptype}"
#                 )
#             if filetype == "idmap" and as_list and ids_key:
#                 map_, duplicates = Dataset._load_map(
#                     os.path.join(self.folder, filename),
#                     as_list=False,
#                     ignore_duplicates=ignore_duplicates,
#                 )
#                 ids = self.load_map(ids_key, as_list=True)
#                 map_ = [map_.get(ids[i], None) for i in range(len(ids))]
#                 nones = map_.count(None)
#                 if nones > 0:
#                     self.config.log(
#                         f"Warning: could not find {nones} ids in map {key}; "
#                         "filling with None."
#                     )
#             else:
#                 map_, duplicates = Dataset._load_map(
#                     os.path.join(self.folder, filename),
#                     as_list=as_list,
#                     ignore_duplicates=ignore_duplicates,
#                 )
#
#             if duplicates > 0:
#                 self.config.log(
#                     f"Warning: map {key} contains {duplicates} duplicate keys, "
#                     "all which have been ignored"
#                 )
#             self.config.log(f"Loaded {len(map_)} keys from map {key}")
#             self._meta[key] = map_
#
#         return self._meta[key]
#
#     def shallow_copy(self):
#         """Returns a data that shares the underlying splits and indexes.
#
#         Changes to splits and indexes are also reflected on this and the copied data.
#         """
#         copy = Dataset(self.config, self.folder)
#         copy._num_entities = self.num_entities()
#         copy._num_relations = self.num_relations()
#         copy._triples = self._triples
#         copy._meta = self._meta
#         copy._indexes = self._indexes
#         copy.index_functions = self.index_functions
#         return copy
#
#     # Access
#     def split(self, split: str) -> Tensor:
#         """Return the split of the specified name.
#
#         If the split is not yet loaded, load it. Returns an Nx3 IntTensor of
#         spo-triples.
#
#         """
#         return self.get(split)
#
#     def train(self) -> Tensor:
#         """Return training split.
#
#         If the split is not yet loaded, load it. Returns an Nx3 IntTensor of
#         spo-triples.
#
#         """
#         return self.split("train")
#
#     def valid(self) -> Tensor:
#         """Return validation split.
#
#         If the split is not yet loaded, load it. Returns an Nx3 IntTensor of
#         spo-triples.
#
#         """
#         return self.split("valid")
#
#     def test(self) -> Tensor:
#         """Return test split.
#
#         If the split is not yet loaded, load it. Returns an Nx3 IntTensor of
#         spo-triples.
#
#         """
#         return self.split("test")
#
#     def num_entities(self) -> int:
#         if not self._num_entities:
#             self._num_entities = len(self.entity_ids())
#         return self._num_entities
#
#     def num_relations(self) -> int:
#         if not self._num_relations:
#             self._num_relations = len(self.relation_ids())
#         return self._num_relations
#
#     def num_timestamps(self) -> int:
#         if not self._num_timestamps:
#             self._num_timestamps = len(self.timestamp_ids())
#         return self._num_timestamps
#
#     def entity_ids(self, indexes: Optional[Union[int, Tensor]] = None) -> Union[str, List[str], np.ndarray]:
#         """Decode indexes to entity ids.
#
#         See `Dataset#map_indexes` for a description of the `indexes` argument.
#         """
#         return self.map_indexes(indexes, "entity_ids")
#
#     def relation_ids(self, indexes: Optional[Union[int, Tensor]] = None) -> Union[str, List[str], np.ndarray]:
#         """Decode indexes to relation ids.
#
#         See `Dataset#map_indexes` for a description of the `indexes` argument.
#         """
#         return self.map_indexes(indexes, "relation_ids")
#
#     def timestamp_ids(self, indexes: Optional[Union[int, Tensor]] = None) -> Union[str, List[str], np.ndarray]:
#         """Decode indexes to relation ids.
#
#         See `Dataset#map_indexes` for a description of the `indexes` argument.
#         """
#         return self.map_indexes(indexes, "timestamp_ids")
#
#     def entity_strings(
#             self, indexes: Optional[Union[int, Tensor]] = None) -> Union[str, List[str], np.ndarray]:
#         """Decode indexes to entity strings.
#
#         See `Dataset#map_indexes` for a description of the `indexes` argument.
#
#         """
#         map_ = self.load_map(
#             "entity_strings",
#             as_list=True,
#             ids_key="entity_ids",
#             ignore_duplicates=True
#         )
#         return self._map_indexes(indexes, map_)
#
#     def relation_strings(
#             self, indexes: Optional[Union[int, Tensor]] = None) -> Union[str, List[str], np.ndarray]:
#         """Decode indexes to relation strings.
#
#         See `Dataset#map_indexes` for a description of the `indexes` argument.
#
#         """
#         map_ = self.load_map(
#             "relation_strings",
#             as_list=True,
#             ids_key="relation_ids",
#             ignore_duplicates=True,
#         )
#         return self._map_indexes(indexes, map_)
#
#     def meta(self, key: str) -> Any:
#         """Return metadata stored under the specified key."""
#         return self._meta[key]
#
#     def index(self, key: str) -> Any:
#         """Return the index stored under the specified key.
#
#         Index means any data structure that is derived from the data, including
#         statistics and indexes.
#
#         If the index has not yet been computed, computes it by calling the function
#         specified in `self.index_functions`.
#
#         See `kge.indexing.create_default_index_functions()` for the indexes available by
#         default.
#
#         """
#         if key not in self._indexes:
#             self.index_functions[key](self)
#         return self._indexes[key]
#
#     @staticmethod
#     def _map_indexes(indexes, values):
#         "Return the names corresponding to specified indexes"
#         if indexes is None:
#             return values
#         elif isinstance(indexes, int):
#             return values[indexes]
#         else:
#             shape = indexes.shape
#             indexes = indexes.view(-1)
#             names = np.array(list(map(lambda i: values[i], indexes)), dtype=str)
#             return names.reshape(shape)
#
#     def map_indexes(
#             self, indexes: Optional[Union[int, Tensor]], key: str) -> Union[Any, List[Any], np.ndarray]:
#         """Maps indexes to values using the specified map.
#
#         `key` refers to the key of a map file of the data, which associates a value
#         with each numerical index. The map file is loaded automatically.
#
#         If `indexes` is `None`, return all values. If `indexes` is an integer, return
#         the corresponding value. If `indexes` is a Tensor, return an ndarray of the same
#         shape holding the corresponding values.
#
#         """
#         map_ = self.load_map(key, as_list=True)
#         return Dataset._map_indexes(indexes, map_)
