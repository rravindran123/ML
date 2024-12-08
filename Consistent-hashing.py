import numpy as np
import hashlib

class consistentHashing():
    def __init__(self, num_replicas=3):
        self.nodes = set()
        self.ring= {}
        self.num_replicas = num_replicas
        self.sorted_keys = []
        self.key_map= {}

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node):
        