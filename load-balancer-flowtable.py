import hashlib
import timeout_decorator

class consistenHash():
    def __init__(self, servers:list, num_replicas=3):
        self.servers = set(servers)
        self.num_replicas = num_replicas
        self.ring={} # Consistent hash ring
        self.sorted_keys=[] # Sorted list of virtual node hash keys
        self.fast_path_expiry={} # Fast path cache  for IP flows
        self.fast_path_cache = {} #Expiry times for fast path entries

        for serv in servers:
            self.add_server(serv)
    
    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)


    def add_server(self, server):
        ''' add a server to the hash ring'''
        self.servers.add(server)
        for i in range(self.num_replicas):
            virtual_node = f"{server}-{i}"
            hashed_key= self._hash(virtual_node)
            self.ring[hashed_key]=server
            self.sorted_keys.append(hashed_key)
        self.sorted_keys.sort()
    
    def remove_server(self, server):
        '''remove a server from the hash ring'''
        self.servers.remove(server)
        for i in range(self.num_replicas):
            virtual_node = f"{server}-{i}"
            hashed_key= self._hash(virtual_node)
            self.ring.pop(hashed_key)
            self.sorted_keys.remove(hashed_key)
    
    def get_server(self, flow):
        if flow in self.fast_path_cache:
            return self.fast_path_cache[flow]
    
    

    