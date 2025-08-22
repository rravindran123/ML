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
        hashed_key = self._hash(flow)
        for key in self.sorted_keys:
            if key >= hashed_key:
                return self.ring[key]
        return self.ring[self.sorted_keys[0]]
    
    def add_fast_path(self, flow, server, expiry_time):
        self.fast_path_cache[flow]=server
        self.fast_path_expiry[flow]=expiry_time 
    
    def remove_fast_path(self, flow):
        self.fast_path_cache.pop(flow)
        self.fast_path_expiry.pop(flow) 
    
    def get_fast_path(self, flow):
        if flow in self.fast_path_cache:
            return self.fast_path_cache[flow]
        return None
    
    def get_fast_path_expiry(self, flow):
        if flow in self.fast_path_expiry:
            return self.fast_path_expiry[flow]
        return None
    
    def list_servers(self):
        print(f"List of servers in the hash ring: {self.servers}")
        print(f"List of virtual nodes: {self.ring}")
        print(f"Sorted keys: {self.sorted_keys}")
        print(f"Fast path cache: {self.fast_path_cache}")
        print(f"Fast path expiry: {self.fast_path_expiry}")
    
    def list_fast_path(self):
        print(f"Fast path cache: {self.fast_path_cache}")
        print(f"Fast path expiry: {self.fast_path_expiry}")

#gemerate test tuples
def generate_test_tuples(num_tuples):
    tuples = []
    for i in range(num_tuples):
        tuples.append((f"192.168.1.{i}", f"192.168.2.{i}", i, i+1, "TCP"))
    return tuples

#generate a key that is the concatenation of the 5 tuple
def generate_key(flow):
    return f"{flow[0]}-{flow[1]}-{flow[2]}-{flow[3]}-{flow[4]}"

# Test the consistent hash class
servers = ['server1', 'server2', 'server3']
ch = consistenHash(servers)
ch.list_servers()
ch.add_server('server4')
ch.list_servers()
ch.remove_server('server1')
ch.list_servers()
#Test flow with 5 tuples - IP source, IP destination, port source, port destination, protocol
flows = generate_test_tuples(5)
for flow in flows:
    print(f"Flow: {flow} Server: {ch.get_server(flow[0])}")
#Add fast path entries
key1 = generate_key(flows[0])
key2 = generate_key(flows[1])
key3 = generate_key(flows[2])
ch.add_fast_path(key1, 'server1', 10)
ch.add_fast_path(key2, 'server2', 20)
ch.add_fast_path(key3, 'server3', 30)
ch.list_fast_path()
print(f"Fast path for flow {key1}: {ch.get_fast_path(key1)}")
print(f"Fast path for flow {key2}: {ch.get_fast_path(key2)}")
print(f"Fast path for flow {key3}: {ch.get_fast_path(key3)}")   
print(f"Fast path expiry for flow {key1}: {ch.get_fast_path_expiry(key1)}")
print(f"Fast path expiry for flow {key2}: {ch.get_fast_path_expiry(key2)}")
print(f"Fast path expiry for flow {key3}: {ch.get_fast_path_expiry(key3)}")
#Remove fast path entries
ch.remove_fast_path(key1)
ch.list_fast_path()
#List servers
ch.list_servers()
#Test flow with 5 tuples - IP source, IP destination, port source, port destination, protocol
flows = generate_test_tuples(5)
for flow in flows:
    print(f"Flow: {flow} Server: {ch.get_server(flow[0])}")
#Remove server
ch.remove_server('server2') 
ch.list_servers()
#Test flow with 5 tuples - IP source, IP destination, port source, port destination, protocol
flows = generate_test_tuples(5)
for flow in flows:
    print(f"Flow: {flow} Server: {ch.get_server(flow[0])}")
#Add server
ch.add_server('server2')
ch.list_servers()
#Test flow with 5 tuples - IP source, IP destination, port source, port destination, protocol
    

    