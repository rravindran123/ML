from dataclasses import dataclass

#@dataclass
class Node:
    def __init__(self, key):
        self.key =  key
        self.counter: int =0

    def __eq__(self, value):
        pass

    def __gt__(self, obj):
        pass

    def __str__(self):
        return f"counter:{self.counter}"

class Data:
    def __init__(self, value):
        self.value: int = value
        self.noderef = None

    def __eq__(self, value):
        pass

    def __gt__(self, obj):
        pass

    def __str__(self):
        return f"value : {self.value}"

class lruCache:
    def __init__(self, capacity=10):
        self.cacheDict: dict= {}
        self.itemCount: int =0
        self.hitList: list= []
        self.cacheCapacity=capacity

    def getItem(self, key):
        # check if the item exists in the dictionary
        value=None
        if key in self.cacheDict:
            value = self.cacheDict[key]
            value.noderef.counter +=1
            #sort the dictionary
            self.hitList.sort(key=lambda Node: Node.counter)
        else:
            print(f"{key} is not in the dictionary")

        return value
    
    def addItem(self, key, value):
        newData = Data(value)
        self.cacheDict[key]=newData
        newNode = Node(key)
        self.hitList.append(newNode)
        self.itemCount +=1
        newData.noderef = newNode

    def insertItem(self, key, value):
        if key in self.cacheDict:
            print(f"{key} already present")
        else:
            if self.itemCount >= self.cacheCapacity:
                print("Case - more than cache capacity")
                #remove the least used item
                lrukey = self.hitList[0].key
                print(f"the list is full, removing {lrukey} and {self.cacheDict[lrukey]}")
                self.cacheDict.pop(lrukey)
                self.hitList.pop(0)
                self.addItem(key,value)
            else:
                self.addItem(key,value)
    
    def printCache(self):
        for k, v in self.cacheDict.items():
            print(f"key : {k}, value: {v}, counter: {v.noderef.counter}")
            

# LRU cache using OrderedDict
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int):
        if key in self.cache:
            # Move accessed item to the end
            self.cache.move_to_end(key)
            return self.cache[key]
        return -1

    def put(self, key: int, value: int):
        if key in self.cache:
            # Update the value and move to the end
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove the least recently used item
            self.cache.popitem(last=False)



def main():
    lru = lruCache(7)
    with open("userdata.txt","r") as file:
        for line in file:
            key,value = line.split(" ")
            value = int(value)
            lru.insertItem(key,value)
    lru.printCache()

    print("\n Accessing items")
    lru.getItem("ert")
    lru.getItem("ert")
    lru.getItem("ert")
    lru.getItem("dfa")
    lru.getItem("sdf")

    lru.printCache()

    lru.insertItem("kgh",1000)

    print("\n After inserting more than capacity")

    lru.printCache()



if __name__=="__main__":
    main()