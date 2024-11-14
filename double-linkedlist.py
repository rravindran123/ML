import os
from dataclasses import dataclass


class data:
    def __init__(self, _name, _salary):
        self.name: str = _name
        self.salary: int = _salary
    
    def __str__(self):
        return f" name: {self.name}, salary: {self.salary}"


class node:
    def __init__(self, _data):
        self.userdata : data = _data
        self.next_item : node = None
        self.prev_item: node = None


class dll:
    def __init__(self):
        self.head:node = None
        self.tail:node = None
    
    #add _data to the end of the list
    def append(self, _data):
        if not self.head:
            newData = node(_data)
            self.head=self.tail= newData
        else:
            newData = node(_data)
            self.tail.next_item = newData
            newData.prev_item = self.tail
            self.tail=newData
    
    def printforward(self):
        if self.head:
            currentNode = self.head
            while currentNode!= None:
                print(f"Node data: {currentNode.userdata}")
                currentNode = currentNode.next_item


    def printbackward(self):
        if self.tail:
            currentNode = self.tail
            while currentNode != None:
                print(f"Node data: {currentNode.userdata}")
                currentNode = currentNode.prev_item

def main():
    doubleLinkedList = dll()

    data1 = data('a', 10000)
    data2 = data('b', 5000)
    data3 = data('c', 3000)

    doubleLinkedList.append(data1)
    doubleLinkedList.append(data2)
    doubleLinkedList.append(data3)

    print("Printing from head")
    doubleLinkedList.printforward()

    print("Printing from tail")
    doubleLinkedList.printbackward()


if __name__== "__main__":
    main()


        
