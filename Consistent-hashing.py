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
        self.nodes.add(node)
        for i in range(self.num_replicas):
            virtual_node_key = f"{node}-{i}"
            key = self._hash(virtual_node_key)
            self.ring[key]=node
            self.sorted_keys.append(key)
        self.sorted_keys.sort()
        
        self._reassign_keys()

    def remove_node(self, node):
        self.nodes.remove(node)
        removed_keys =[]
        for i in range(self.num_replicas):
            virtual_node_key= f"{node}-{i}"
            hashed_key = self._hash(virtual_node_key)
            removed_keys.append(hashed_key)
            del self.ring[hashed_key]
            self.sorted_keys.remove(hashed_key)
        
        self._reassign_keys()

    def _reassign_keys(self):
        reassigned_map={}
        for key, current_node in self.key_map.items():
            new_node = self.get_node(key)
            if new_node != current_node:
                print(f"Key {key} reassigned from {current_node} to {new_node}")
            reassigned_map[key]=new_node

        self.key_map= reassigned_map

    def get_node(self, key):
        if not self.ring:
            return None
        
        hashed_key = self._hash(key)
        for node_key in self.sorted_keys:
            if hashed_key <= node_key:
                return self.ring[node_key]
        return self.ring[self.sorted_keys[0]]
    
    def assign_key(self, key):
        """Assign a key to the appropriate node based on consistent hashing."""
        node = self.get_node(key)
        self.key_map[key] = node
        print(f"Key {key} is assigned to node {node}")

    def display_ring(self):
        """Display the current state of the hash ring."""
        print("Hash Ring (virtual nodes):")
        for hashed_key in self.sorted_keys:
            print(f"Hash: {hashed_key}, Node: {self.ring[hashed_key]}")

    def display_keys(self):
        """Display the current key-to-node mappings."""
        print("Key-to-Node Mappings:")
        for key, node in self.key_map.items():
            print(f"Key: {key}, Node: {node}")


# Example usage
if __name__ == "__main__":
    ch = consistentHashing(num_replicas=3)

    # Add nodes
    ch.add_node("Node1")
    ch.add_node("Node2")
    ch.add_node("Node3")
    print("\n--- Initial Hash Ring ---")
    ch.display_ring()

    # Assign keys
    ch.assign_key("key1")
    ch.assign_key("key2")
    ch.assign_key("key3")
    ch.assign_key("key4")
    ch.display_keys()

    # Remove a node
    print("\n--- Removing Node2 ---")
    ch.remove_node("Node1")
    ch.display_ring()
    ch.display_keys()
    


