import heapq
from typing import List, Optional, Union
import collections

import torch

class LRUCache:

    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = collections.OrderedDict()
        self.total = 0
        self.hit = 0
        for idx, i in enumerate(range(self.capacity)):
            self.put(i, idx)

    def get(self, key):
        self.total += 1
        if key not in self.queue:
            return -1
        self.hit +=1
        value = self.queue.pop(key)
        self.queue[key] = value
        return self.queue[key]
    
    def in_cache(self, key):
        if key in self.queue:
            return True
        return False

    def get_index(self):
        if len(self.queue.items()) == self.capacity:
            return self.queue.popitem(last=False)
        return None

    def put(self, key, value):
        if key in self.queue:
            self.queue.pop(key)
        self.queue[key] = value

class HotVocabTable:
    def __init__(self, num_dynamic_tokens=256):
        self.cache = LRUCache(num_dynamic_tokens)
        self.token_ids = torch.tensor([k for k in self.cache.queue.keys()], dtype=torch.int32, device="cuda")
        self.id_map = {'gather':[], 'scatter':[]}


    def add_token(self, token_ids: Union[torch.Tensor, List[torch.Tensor]]) -> None:
        if not isinstance(token_ids, list):
            token_ids = [token_ids]
        
        self.id_map['scatter'].clear()
        self.id_map['gather'].clear()

        for t in token_ids:
            if t.dim() != 1:
                t = t.flatten()
            
            t = torch.unique(t)
            for item in t:
                if self.cache.in_cache(item):
                    self.cache.get(item)
                    continue

                idx = self.cache.get_index()
                if idx is not None:
                    self.id_map['scatter'].append(idx[1])
                    self.id_map['gather'].append(item)
                    self.cache.put(i, idx[1])
                else:
                    pass
        
    def update_table(self, hot_table, ori_table):
        data = ori_table[self.id_map['gather']]
        hot_table[self.id_map['scatter']] = data

    def get_hot_token_ids(self):
        return self.token_ids
