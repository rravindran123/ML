# copilot: disable

import hashlib

class HashTable:
    def __init__(self):
        self.table_size =0
        self.table = {}
    
    def insert(self, key, value):
        hash_key = self.getHash(key)
        print(f"key {key}, hash_key : {hash_key}")
        if hash_key not in self.table:
            self.table[hash_key] = [value]
        else:
            self.table[hash_key].append(value)

    def removeValue(self,key, value):
        hash_key = self.getHash(key)
        if hash_key in self.table:
            self.table[hash_key].remove(value)

    def removeKey(self, key):
        hash_key = self.getHash(key)
        if hash_key in self.table:
            del self.table[hash_key]    

    def getHash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def __str__(self):
        items = []
        for key, value in self.table.items():
            for i in range(len(value)):
                items.append(f"key: {key}, value:{value[i]}")
        return "\n".join(items)


def main():
    table = HashTable()
    table.insert("ravi", 100)
    table.insert("adfadfadf", 200)
    table.insert("234sdfd", 300)
    print(table)

    print("\n\n after remvoing")

    table.removeKey("ravi")
    print(table)

if __name__ == "__main__":
    main() 