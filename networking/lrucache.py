import os
from os import utime
import torch


# class node:
#     def __init__(self, value, objsize):
#         self.data= value
#         self.objsize = objsize
#         self.req =0

#     def __repr__(self):
#         return f"value: {self.data}, size: {self.objsize}, #ofreq: {self.req}"
        

# class lrucache:
#     def __init__(self, maxsize):
#         self.cacheCapacity = maxsize
#         self.cacheSize=0
#         self.cache = {}

#     def put(self, key:str,  item:node):
#         print(f"putting item {item} with key {key} in the cache")
#         if key in self.cache:
#             print("key already in the cache")
#             return
#         else:
#             if item.objsize + self.cacheSize <= self.cacheCapacity:
#                 self.cache[key] = item
#                 self.cacheSize += item.objsize
#                 #sort the dictionary using the 
#             else:
#                 #evict the least used objects to accomodate the new size
#                 print("size exceeded, evicting the least used objects")
#                 sizediff = item.objsize + self.cacheSize - self.cacheCapacity
#                 print(f"sizediff {sizediff}")
#                 while sizediff >0:
#                     last_key, last_item = list(self.cache.items())[-1]
#                     obj = self.cache[last_key]
#                     print(f"last item key :{last_key}, {last_item}, {obj.objsize}")
#                     self.cache.pop(last_key)
#                     sizediff -= item.objsize
#                     print(f"new sizediff {sizediff}")
                    
#                 self.cache[key] = item
#                 self.cacheSize += item.objsize
#                 #sort the dictionary
#                 #sself.sort_cache_by_req()

#     #sort the dictionary
#     def sort_cache_by_req(self):
#         self.cache = dict(sorted(self.cache.items(), key=lambda item:item[1].req, reverse=True))

#     # takes the key and returns the 
#     def get(self, key):
#         if key in self.cache:
#             obj =  self.cache[key]
#             obj.req += 1
#             self.sort_cache_by_req()
#             return obj
#         else:
#             return None

#     def __repr__(self):
#         returnstr = f"printing the cache \n"
#         for key, values in self.cache.items():
#             returnstr += f"key: {key}, {values}\n"
#         return returnstr


# obj1 = node("adfdafa", 100)
# obj2 = node("were", 100)
# obj3= node("weres", 100)
# obj4 = node("quick", 100)

# lruc = lrucache(300)
# lruc.put("ravi", obj1)
# print(lruc)
# lruc.put("akele", obj2)
# print(lruc)
# lruc.put("crab", obj3)
# print(lruc)

# print(lruc.get("ravi"))
# print(lruc.get("ravi"))
# print(lruc.get("crab"))

# lruc.put("ok",  obj4)

# print(lruc)
from dataclasses import dataclass

class node:
    def __init__(self, key:str="none", value:int=9999, size:int=9999):
        self.key = key
        self.value = value
        self.size = size
        self.prev = None
        self.next = None

    def __repr__(self):
        return f"key: {self.key}, value :{self.value}, size:{self.size}"

class dll:
    def __init__(self):
        self.head = node()
        self.tail= node()
        self.head.next= self.tail
        self.tail.prev = self.head

    #insert to the head
    def insert(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def delete_from_head(self):
        if self.head.next != None:
            print("The dll has elements")
            noderef = self.head.next
            self.head = noderef.next
            noderef.next.prev = self.head
        else:
            print("the list is empty")

    def get_item_from_tail(self):
        if self.tail.prev != None and self.tail.prev != self.head:
            return self.tail.prev
        else:
            return None

    def move_item_to_head(self, obj):
        if obj.prev == self.head:
            print("already the first node")
        else:
            print("moving the node to the head")
            obj.prev.next = obj.next
            obj.next.prev = obj.prev
            obj.next = self.head.next
            obj.prev = self.head
            self.head.next.prev = obj
            self.head.next = obj
            

    def delete_from_tail(self):
        if self.tail.prev != None:
            print("The dll is not empty")
            noderef = self.tail.prev
            noderef.prev.next = self.tail
            self.tail.prev = noderef.prev
        else:
            print("the list is empty")
        
    def __repr__(self):
        returnlist = f"dll list\n"
        noderef = self.head.next
        while noderef.next != None:
            returnlist += f"node key {noderef.key}, value {noderef.value}\n"
            print(f"node key {noderef.key}, value {noderef.value}\n")
            noderef = noderef.next
        return returnlist

class lruClass:
    def __init__(self, maxSize:int):
        self.cachesize = maxSize
        self.index = dict()
        self.lruitems = dll()
        self.currSize =0

    def insertitem(self, key:str, value:int, objsize:int):

        if self.currSize + objsize <= self.cachesize:
            obj = node(key, value, objsize)
            self.lruitems.insert(obj)
            self.currSize += objsize
            self.index[key]=obj
        else:
            #delete the last item in the dll
            sizediff = self.currSize + objsize - self.cachesize
            while sizediff >0:
                lruitem = self.lruitems.get_item_from_tail()
                print(f" size of lruitem: {lruitem.size}")
                sizediff -= lruitem.size
                self.currSize -= lruitem.size
            obj = node(key, value, objsize)
            self.lruitems.insert(obj)
            self.currSize += objsize
            self.index[key]=obj

    def getitem(self, key):
        if key in self.index:
            print("Key in the lru cache")
            #move the item to the head
            self.lruitems.move_item_to_head(self.index[key])
            return self.index[key]
        else:
            print("key doesnt exist in the list")
            return None

    def __repr__(self):
        returnDat = f"printing the list"
        returnDat += f"{self.lruitems}"
        return returnDat

#test the code
lru= lruClass(300)
lru.insertitem("1", 2343, 100)
lru.insertitem("2", 454, 100)
lru.insertitem("3", 67676, 100)
lru.insertitem("4", 67676, 100)


print(lru)
#get items
print(lru.getitem("1"))

print(lru)
