# coding=utf-8
"""
Utilities to profile the execution time
"""

from time import time


class TimeCounter:
    def __init__(self):
        self.temp_time = 0
        self.acum_time = 0
        self.n_starts = 0

    def reset(self):
        self.temp_time = 0
        self.acum_time = 0
        self.n_starts = 0

    def start(self):
        if self.n_starts == 0:
            self.temp_time = time()
        self.n_starts += 1

    def stop(self):
        if self.n_starts == 1:
            self.acum_time += time() - self.temp_time
        if self.n_starts > 0:
            self.n_starts -= 1

    def get_time(self):
        return self.acum_time


class TimeCounterCollection:
    def __init__(self, list_names):
        self.tc_collection = dict((name, TimeCounter()) for name in list_names)

    def reset(self, name):
        self.tc_collection[name].reset()

    def reset_all(self):
        for tc in self.tc_collection.values():
            tc.reset()

    def start(self, name):
        self.tc_collection[name].start()

    def stop(self, name):
        self.tc_collection[name].stop()

    def print_times(self):
        for name, time_counter in self.tc_collection.items():
            print(name, '->', time_counter.get_time())
