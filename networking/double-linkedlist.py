import os
from dataclasses import dataclass


class data:
    def __init__(self, _name, _salary):
        self.name: str = _name
        self.salary: int = _salary
    
    def __str__(self):
        return f" name: {self.name}, salary: {self.salary}"

    def __eq__(self, obj):
        return (self.name==obj.name and self.salary==obj.salary)

class node:
    def __init__(self, _data):
        self.userdata : data = _data
        self.next_item : node = None
        self.prev_item: node = None


class dll:
    def __init__(self):
        self.head:node = None
        self.tail:node = None
        self.numItems=0
    
    #add _data to the end of the list
    def append(self, _data):
        if not self.head:
            newData = node(_data)
            self.head=self.tail= newData
            self.numItems+=1
        else:
            newData = node(_data)
            self.tail.next_item = newData
            newData.prev_item = self.tail
            self.tail=newData
            self.numItems+=1
    
    def popleft(self):
        assert self.head != None, "List is empty, cannot pop from left"
        old_head = self.head
        if self.head.next_item != None:
            self.head.next_item.prev_item= None
            self.head=self.head.next_item
            self.numItems -=1
        else:
            self.head= self.tail= None
            self.numItems -=1

        old_head.next_item= None
        old_head.prev_item= None
        return old_head

    def remove(self, user):
        assert self.head != None, "List is empty"
        currentptr = self.head
        removedItem = None

        while currentptr != None:
            if currentptr.userdata == user:
                removedItem = currentptr
                if self.numItems==1:
                    print("Matched only item in the list..")
                    self.head= self.tail= None
                    self.numItems -=1
                    break
                elif self.numItems==2:
                    if currentptr.prev_item == None:
                        self.head = currentptr.next_item
                        currentptr.next_item.prev_item=None
                        currentptr.next_item= None
                        self.numItems -=1
                        break
                    elif currentptr.next_item == None:
                        self.tail = currentptr.prev_item
                        currentptr.prev_item.next_item= None
                        currentptr.prev_item = None
                        self.numItems -=1
                        break
                else:
                    currentptr.prev_item.next_item=currentptr.next_item
                    currentptr.next_item.prev_item=currentptr.prev_item
                    self.numItems -=1
                    break
            else:
                currentptr=currentptr.next_item
        return removedItem

    def printforward(self):
        if self.numItems ==0:
            print("List is empty")
        elif self.head:
            currentNode = self.head
            while currentNode!= None:
                print(f"Node data: {currentNode.userdata}")
                currentNode = currentNode.next_item


    def printbackward(self):
        if self.numItems ==0:
            print("List is empty")
        elif self.tail:
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

    #doubleLinkedList.popleft()
    doubleLinkedList.remove(data2)
    doubleLinkedList.remove(data3)
    doubleLinkedList.remove(data1)

    print("dll after removing item")
    doubleLinkedList.printforward()

    doubleLinkedList.append(data1)
    doubleLinkedList.append(data2)
    doubleLinkedList.append(data3)

    # print("Printing from tail")
    doubleLinkedList.printbackward()

    #add user data reading from a file
    with open("userdata.txt", 'r') as file:
        for line in file:
            name, salary = line.split(" ")
            salary = int(salary)
            data_instance = data(name, salary)
            doubleLinkedList.append(data_instance)

    # print("Printing from tail")
    doubleLinkedList.printbackward()




if __name__== "__main__":
    main()


        
