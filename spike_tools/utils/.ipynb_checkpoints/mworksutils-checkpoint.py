from __future__ import division, print_function, unicode_literals
import sqlite3
import zlib

import numpy as np
import pandas as pd

import msgpack

try:
    buffer
except NameError:
    # Python 3
    buffer = bytes


class MWK2Reader(object):

    _compressed_text_type_code = 1
    _compressed_msgpack_stream_type_code = 2

    def __init__(self, filename):
        self._conn = sqlite3.connect(filename)
        self._unpacker = msgpack.Unpacker(raw=False, strict_map_key=False)

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    @staticmethod
    def _decompress(data):
        return zlib.decompress(data, -15)

    def __iter__(self):
        for code, time, data in self._conn.execute('SELECT * FROM events'):
            if not isinstance(data, buffer):
                yield (code, time, data)
            else:
                try:
                    obj = msgpack.unpackb(data, raw=False)
                except msgpack.ExtraData:
                    # Multiple values, so not valid compressed data
                    pass
                else:
                    if isinstance(obj, msgpack.ExtType):
                        if obj.code == self._compressed_text_type_code:
                            yield (code,
                                   time,
                                   self._decompress(obj.data).decode('utf-8'))
                            continue
                        elif (obj.code ==
                              self._compressed_msgpack_stream_type_code):
                            data = self._decompress(obj.data)
                self._unpacker.feed(data)
                try:
                    while True:
                        yield (code, time, self._unpacker.unpack())
                except msgpack.OutOfData:
                    pass
                
                
                
def get_trial_indices(events, df = False, delay_sec= 1):
    if df:
        times = np.array([row.time for i, row in events.iterrows()])
    else:
        times = np.array([i.time for i in events])

    diff_times = np.diff(times)
    trials = []
    mini_trial = [0]
    print( diff_times)
    for i, t in enumerate(diff_times):
        if t < delay_sec*1e6:
            mini_trial.append(i+1)
        else:
            trials.append(mini_trial)
            mini_trial = [i+1]
    trials.append(mini_trial)
    print(i, len(trials))
    return trials

