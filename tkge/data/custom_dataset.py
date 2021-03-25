import torch
from torch.utils.data.dataset import Dataset as PTDataset
import numpy as np

from typing import Dict, List, Tuple, Optional
import enum
import datetime
import time

from tkge.data.dataset import DatasetProcessor
from tkge.common.config import Config

from collections import defaultdict

SPOT = enum.Enum('spot', ('s', 'p', 'o', 't'))


@DatasetProcessor.register(name="icews14_atise")
class ICEWS14AtiseDatasetProcessor(DatasetProcessor):
    def process(self):
        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts_id])
            self.train_set['timestamp_float'].append([ts])

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.valid_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([ts_id])
            self.valid_set['timestamp_float'].append([ts])

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.test_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([ts_id])
            self.test_set['timestamp_float'].append([ts])

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

    def process_time(self, origin: str):
        # TODO (gengyuan) move to init method
        self.gran = self.config.get("dataset.temporal.gran")

        start_sec = time.mktime(time.strptime('2014-01-01', '%Y-%m-%d'))

        end_sec = time.mktime(time.strptime(origin, '%Y-%m-%d'))
        day = int((end_sec - start_sec) / (self.gran * 24 * 60 * 60))

        return day


@DatasetProcessor.register(name="yago11k")
class Yago11kDatasetProcessor(DatasetProcessor):
    def process(self):
        year_list = []

        for rd in self.train_raw:
            head, rel, tail, ts_start, ts_end = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_start = self.process_time(ts_start, True)
            ts_end = self.process_time(ts_start, False)

            if ts_start < ts_end:
                ts_end = self.config.get("dataset.args.year_max")

        for rd in self.valid_raw:
            pass

        for rd in self.test_raw:
            pass

    def process_time(self, origin: str, start: bool = True):
        year = origin.split('-')[0]

        if year.find('#') != -1 and len(year) == 4:
            year = int(year)
        else:
            year = self.config.get("dataset.args.year_min") if start else self.config.get("dataset.args.year_max")

        return year


@DatasetProcessor.register(name="yago15k_TA")
class YAGO15KDatasetProcessor(DatasetProcessor):
    def process(self):
        """
        Processes the raw data for each data type (i.e. train, valid and test) of the YAGO15k dataset.
        Uses the resulting timespan of the function process_time to add as many quadruples to each data set as years
        where that fact is true.
        """
        self.tem_dict = {
            '0y': 0, '1y': 1, '2y': 2, '3y': 3, '4y': 4, '5y': 5, '6y': 6, '7y': 7, '8y': 8, '9y': 9,
            '01m': 10, '02m': 11, '03m': 12, '04m': 13, '05m': 14, '06m': 15, '07m': 16, '08m': 17, '09m': 18,
            '10m': 19, '11m': 20, '12m': 21,
            '0d': 22, '1d': 23, '2d': 24, '3d': 25, '4d': 26, '5d': 27, '6d': 28, '7d': 29, '8d': 30, '9d': 31,
        }

        data_splits = ["train", "valid", "test"]
        data_raw_mappings = {data_splits[0]: self.train_raw, data_splits[1]: self.valid_raw,
                             data_splits[2]: self.test_raw}
        data_set_mappings = {data_splits[0]: self.train_set, data_splits[1]: self.valid_set,
                             data_splits[2]: self.test_set}

        for data_split in data_splits:
            index = 0
            while index < len(data_raw_mappings[data_split]):
                rd = data_raw_mappings[data_split][index]
                fact = rd.strip().split('\t')
                # self.config.log(f"Processing fact in line {index + 1}: {fact}")

                triple = fact[0][1:-1], fact[1][1:-1], fact[2][1:-1]
                head_id = self.index_entities(triple[0])
                rel_id = self.index_relations(triple[1])
                tail_id = self.index_entities(triple[2])

                year_start, year_stop, index = self.process_time(data_split, index=index, fact=fact, triple=triple)

                for y in range(year_start, year_stop + 1):
                    ts_id = self.index_timestamps(y)
                    ts_float = []
                    ts_float.extend([self.tem_dict[f'{int(yi):01}y'] for yi in str(y)])

                    data_set_mappings[data_split]['triple'].append([head_id, rel_id, tail_id])
                    data_set_mappings[data_split]['timestamp_id'].append([ts_id])
                    data_set_mappings[data_split]['timestamp_float'].append(ts_float)

                    self.all_triples.append([head_id, rel_id, tail_id])
                    self.all_quadruples.append([head_id, rel_id, tail_id, ts_id])

                    # self.config.log(f" {data_split} tuple: {str([head_id, rel_id, tail_id, ts_id, ts_float])}")

                index += 1

    def process_time(self, origin: str, index: int = 0, fact: List[str] = None, triple: Tuple = None):
        """
        Processes the time information in the YAGO15k dataset.
        Since there are not only timestamps but also time modifiers, namely 'occursSince' and 'occursUntil', and
        corresponding tuples that appear twice in a row, a few cases need to be addressed:
        - Case 1: The timespan is given by the data.
        - Case 2: The timespan is not given explicitly, but with argument-less modifiers that indicate universal truth.
        - Case 3: The timespan is partly given by the data, i.e. the start time is known.
        - Case 4: The timespan is partly given by the data, i.e. the end time is known.
        - Case 5: There is no time information given at all.
        In cases 1 and 2 the index needs to be incremented by 1 because the next line was already processed then.
        Returns the start and end timestamp (only as the year) of the triple, as well as the maybe modified index.
        """
        data = {"train": self.train_raw, "valid": self.valid_raw, "test": self.test_raw}
        data_raw = data[origin]
        valid_temp_mods = ["occursSince", "occursUntil"]

        # process time of the current fact
        temp_mod = fact[3][1:-1] if len(fact) >= 4 and fact[3][1:-1] in valid_temp_mods else None
        year = int(fact[4].split('-')[0][1:]) if len(fact) == 5 else 0

        # process time of the next fact only if there exists at least one more fact after this
        if index + 1 < len(data_raw):
            next_fact = data_raw[index + 1].strip().split('\t')
            next_triple = next_fact[0][1:-1], next_fact[1][1:-1], next_fact[2][1:-1]

            next_temp_mod = next_fact[3][1:-1] if len(next_fact) >= 4 and next_fact[3][1:-1] in valid_temp_mods else None
            next_year = int(next_fact[4][1:].split('-')[0]) if len(next_fact) == 5 else 0

            is_closed_timespan = triple == next_triple and temp_mod and year != 0 and next_temp_mod and next_year != 0
        else:
            next_temp_mod = None
            next_year = 0

            is_closed_timespan = False

        if is_closed_timespan:
            # case 1: same triple appears twice in a row and has a temporal modifier as well as a timestamp
            year_stop = next_year
            index += 1
        elif temp_mod and next_temp_mod and year == 0 and next_year == 0:
            # case 2: same triple appears twice in a row but has only a temporal modifier and no timestamp
            year = int(self.config.get('dataset.args.year_min'))
            year_stop = int(self.config.get('dataset.args.year_max'))
            index += 1
        elif temp_mod and temp_mod == 'occursSince':
            # case 3: triple appears only once and is true until now (or forever from the start year)
            year_stop = int(self.config.get('dataset.args.year_max'))
        elif temp_mod and temp_mod == 'occursUntil':
            # case 4: triple appears only once and was true until a certain year
            year_stop = year
            year = int(self.config.get('dataset.args.year_min'))
        else:
            # case 5: otherwise there is no timespan, i.e. a stand alone triple
            year_stop = year

        return year, year_stop, index


@DatasetProcessor.register(name="icews14_TA")
class ICEWS14TADatasetProcessor(DatasetProcessor):
    def process(self):
        self.tem_dict = {
            '0y': 0, '1y': 1, '2y': 2, '3y': 3, '4y': 4, '5y': 5, '6y': 6, '7y': 7, '8y': 8, '9y': 9,
            '01m': 10, '02m': 11, '03m': 12, '04m': 13, '05m': 14, '06m': 15, '07m': 16, '08m': 17, '09m': 18,
            '10m': 19, '11m': 20, '12m': 21,
            '0d': 22, '1d': 23, '2d': 24, '3d': 25, '4d': 26, '5d': 27, '6d': 28, '7d': 29, '8d': 30, '9d': 31,
        }

        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts_id])
            self.train_set['timestamp_float'].append(ts)

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.valid_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.valid_set['triple'].append([head, rel, tail])
            self.valid_set['timestamp_id'].append([ts_id])
            self.valid_set['timestamp_float'].append(ts)

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.test_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts_id = self.index_timestamps(ts)
            ts = self.process_time(ts)

            self.test_set['triple'].append([head, rel, tail])
            self.test_set['timestamp_id'].append([ts_id])
            self.test_set['timestamp_float'].append(ts)

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

    def process_time(self, origin: str):
        ts = []
        year, month, day = origin.split('-')

        ts.extend([self.tem_dict[f'{int(yi):01}y'] for yi in year])
        ts.extend([self.tem_dict[f'{int(month):02}m']])
        ts.extend([self.tem_dict[f'{int(di):01}d'] for di in day])

        return ts


# Deprecated: dataset used for debugging tcomplex training
@DatasetProcessor.register(name="icews14_tcomplex")
class TestICEWS14DatasetProcessor(DatasetProcessor):
    def __init__(self, config: Config):
        super().__init__(config)

        print('==========')

        self.folder = self.config.get("dataset.folder")
        self.level = self.config.get("dataset.temporal.level")
        self.index = self.config.get("dataset.temporal.index")
        self.float = self.config.get("dataset.temporal.float")

        self.reciprocal_training = self.config.get("task.reciprocal_relation")
        # self.filter_method = self.config.get("data.filter")

        self.train_raw: List[str] = []
        self.valid_raw: List[str] = []
        self.test_raw: List[str] = []

        mapping = torch.load('/mnt/data1/ma/gengyuan/baseline/tkbc/tkbc/mapping.pt')

        self.ent2id = mapping[0]
        self.rel2id = mapping[1]
        self.ts2id = mapping[2]

        temp = dict()

        for k in self.rel2id.keys():
            temp[k + '(RECIPROCAL)'] = self.rel2id[k] + 230

        self.rel2id.update(temp)

        self.train_set = defaultdict(list)
        self.valid_set = defaultdict(list)
        self.test_set = defaultdict(list)

        self.all_triples = []
        self.all_quadruples = []

        self.load()
        self.process()
        self.filter()

    def load(self):
        train_file = self.folder + "/train.txt"
        valid_file = self.folder + "/valid.txt"
        test_file = self.folder + "/test.txt"

        print('ygygyg')

        with open(train_file, "r") as f:
            if self.reciprocal_training:
                lines = f.readlines()

                for line in lines:
                    self.train_raw.append(line)

                    # for line in lines:

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

    # def index_relations(self, rel: str):
    #     if rel.endswith('(RECIPROCAL)'):
    #         print(rel)
    #         return self.rel2id[rel[:-12]] + 230
    #     else:
    #         return self.rel2id[rel]


# Deprecated: dataset created with timespans instead of concatenated relation and temporal modifier
@DatasetProcessor.register(name="yago15k_timespan")
class YAGO15KDatasetProcessor(DatasetProcessor):
    def process(self):
        """
        Processes the raw data for each data type (i.e. train, valid and test) of the YAGO15k dataset.
        Uses the resulting timespan of the function process_time to add as many quadruples to each data set as years
        where that fact is true.
        """
        data_splits = ["train", "valid", "test"]
        data_raw_mappings = {data_splits[0]: self.train_raw, data_splits[1]: self.valid_raw, data_splits[2]: self.test_raw}
        data_set_mappings = {data_splits[0]: self.train_set, data_splits[1]: self.valid_set, data_splits[2]: self.test_set}

        for data_split in data_splits:
            index = 0
            while index < len(data_raw_mappings[data_split]):
                rd = data_raw_mappings[data_split][index]
                fact = rd.strip().split('\t')

                triple = fact[0][1:-1], fact[1][1:-1], fact[2][1:-1]
                head_id = self.index_entities(triple[0])
                rel_id = self.index_relations(triple[1])
                tail_id = self.index_entities(triple[2])

                year_start, year_stop, index = self.process_time(data_split, index=index, fact=fact, triple=triple)

                for y in range(year_start, year_stop + 1):
                    ts_id = self.index_timestamps(y)

                    data_set_mappings[data_split]['triple'].append([head_id, rel_id, tail_id])
                    data_set_mappings[data_split]['timestamp_id'].append([ts_id])
                    data_set_mappings[data_split]['timestamp_float'].append([y])

                    self.all_triples.append([head_id, rel_id, tail_id])
                    self.all_quadruples.append([head_id, rel_id, tail_id, ts_id])

                index += 1

    def process_time(self, origin: str, index: int = 0, fact: List[str] = None, triple: Tuple = None):
        """
        Processes the time information in the YAGO15k dataset.
        Since there are not only timestamps but also time modifiers, namely 'occursSince' and 'occursUntil', and
        corresponding tuples that appear twice in a row, a few cases need to be addressed:
        - Case 1: The timespan is given by the data.
        - Case 2: The timespan is not given explicitly, but with argument-less modifiers that indicate universal truth.
        - Case 3: The timespan is partly given by the data, i.e. the start time is known.
        - Case 4: The timespan is partly given by the data, i.e. the end time is known.
        - Case 5: There is no time information given at all.
        In cases 1 and 2 the index needs to be incremented by 1 because the next line was already processed then.
        Returns the start and end timestamp (only as the year) of the triple, as well as the maybe modified index.
        """
        data = {"train": self.train_raw, "valid": self.valid_raw, "test": self.test_raw}
        data_raw = data[origin]
        valid_temp_mods = ["occursSince", "occursUntil"]

        # process time of the current fact
        temp_mod = fact[3][1:-1] if len(fact) >= 4 and fact[3][1:-1] in valid_temp_mods else None
        year = int(fact[4].split('-')[0][1:]) if len(fact) == 5 else 0

        # process time of the next fact only if there exists at least one more fact after this
        if index + 1 < len(data_raw):
            next_fact = data_raw[index + 1].strip().split('\t')
            next_triple = next_fact[0][1:-1], next_fact[1][1:-1], next_fact[2][1:-1]

            next_temp_mod = next_fact[3][1:-1] if len(next_fact) >= 4 and next_fact[3][1:-1] in valid_temp_mods else None
            next_year = int(next_fact[4][1:].split('-')[0]) if len(next_fact) == 5 else 0

            is_closed_timespan = triple == next_triple and temp_mod and year != 0 and next_temp_mod and next_year != 0
        else:
            next_temp_mod = None
            next_year = 0

            is_closed_timespan = False

        if is_closed_timespan:
            # case 1: same triple appears twice in a row and has a temporal modifier as well as a timestamp
            year_stop = next_year
            index += 1
        elif temp_mod and next_temp_mod and year == 0 and next_year == 0:
            # case 2: same triple appears twice in a row but has only a temporal modifier and no timestamp
            year = int(self.config.get('dataset.args.year_min'))
            year_stop = int(self.config.get('dataset.args.year_max'))
            index += 1
        elif temp_mod and temp_mod == 'occursSince':
            # case 3: triple appears only once and is true until now (or forever from the start year)
            year_stop = int(self.config.get('dataset.args.year_max'))
        elif temp_mod and temp_mod == 'occursUntil':
            # case 4: triple appears only once and was true until a certain year
            year_stop = year
            year = int(self.config.get('dataset.args.year_min'))
        else:
            # case 5: otherwise there is no timespan, i.e. a stand alone triple
            year_stop = year

        return year, year_stop, index
