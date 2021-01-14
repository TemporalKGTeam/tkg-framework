# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import numpy as np
from scripts import shredFacts
from measure import Measure


class Tester:
    def __init__(self, dataset, model, valid_or_test):
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.valid_or_test = valid_or_test
        self.measure = Measure()

    def getRank(self, sim_scores):  # assuming the test fact is the first one
        # print(sim_scores.shape)

        return (sim_scores > sim_scores[0]).sum(0).float() + 1.

    def replaceAndShred(self, fact, raw_or_fil, head_or_tail):
        head, rel, tail, years, months, days = fact
        if head_or_tail == "head":
            ret_facts = [(i, rel, tail, years, months, days)
                         for i in range(self.dataset.numEnt())]
        if head_or_tail == "tail":
            ret_facts = [(head, rel, i, years, months, days)
                         for i in range(self.dataset.numEnt())]

        filt = list(set(ret_facts) & self.dataset.all_facts_as_tuples)
        filt = [f[0] if head_or_tail == 'head' else f[2] for f in filt]
        # print(filt)
        # assert False

        if raw_or_fil == "raw":
            ret_facts = [tuple(fact)] + ret_facts
        elif raw_or_fil == "fil":
            ret_facts = [tuple(fact)] + list(set(ret_facts) -
                                             self.dataset.all_facts_as_tuples)

        heads, rels, tails, years, months, days = shredFacts(np.array(ret_facts))

        return heads, rels, tails, years, months, days, filt

    def test(self):
        with torch.no_grad():
            rank_list = {'head': [],
                         'tail': []}
            filt_list = {'head': [],
                         'tail': []}
            score_list = {'head': [],
                          'tail': []}

            for i, fact in enumerate(self.dataset.data[self.valid_or_test]):
                print(i)

                settings = ["fil"]
                for raw_or_fil in settings:
                    for head_or_tail in ["head", "tail"]:
                        heads, rels, tails, years, months, days, filt = self.replaceAndShred(
                            fact, raw_or_fil, head_or_tail)

                        sim_scores = self.model(
                            heads, rels, tails, years, months, days)#.cpu().data.numpy()
                        rank = self.getRank(sim_scores)

                        # rank_list[head_or_tail].append(rank)
                        # filt_list[head_or_tail].append(filt)
                        # score_list[head_or_tail].append(sim_scores[0])

                        self.measure.update(rank, raw_or_fil)

        # torch.save(rank_list, '/home/gengyuan/workspace/tkge/tests/desimple_master/ranklist_source.pt')
        # torch.save(filt_list, '/home/gengyuan/workspace/tkge/tests/desimple_master/filtlist_source.pt')
        # torch.save(score_list, '/home/gengyuan/workspace/tkge/tests/desimple_master/scorelist_source.pt')

        self.measure.print_()
        print("~~~~~~~~~~~~~")
        print(len(self.dataset.data[self.valid_or_test]))
        self.measure.normalize(len(self.dataset.data[self.valid_or_test]))
        self.measure.print_()

        return self.measure.mrr["fil"]


class MockTester(Tester):
    def __init__(self):
        self.measure = None

    def test(self):
        torch.manual_seed(0)

        # sim_queries = torch.cat((torch.randint(vocab_size, (query_size, 1)), torch.randint(vocab_size, (query_size, 1)), torch.randint(vocab_size, (query_size, 1))), 1).int()

        settings = ['raw']

        query_size = 1000
        vocab_size = 1000

        for raw_or_fil in settings:
            for head_or_tail in ["head"]:
                torch.manual_seed(0)

                self.measure = Measure()
                random_scores = torch.rand((query_size, vocab_size))

                print(f"current settings {head_or_tail} + {raw_or_fil}")
                for fact in random_scores:
                    rank = self.getRank(fact)
                    self.measure.update(rank, raw_or_fil)

                self.measure.normalize(query_size)
                self.measure.print_()


if __name__ == "__main__":
    MockTester().test()
