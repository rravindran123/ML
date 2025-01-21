import os
from dataclasses import dataclass

class data:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
    
    def __eq__(self, value):
        return (self.name == value.name and self.salary== value.salary)
    
    def __le__(self, value):
        return (self.salary <= value.salary)
    
    def __ge__(self, value):
        return (self.salary >= value.salary)
    
    def __str__(self):
        return f"name: {self.name} and salar: {self.salary}"
    
    def __repr__(self):
        return f"name: {self.name} and salar: {self.salary}"
    
class node:
    def __init__(self, data):
        self.userdata: data = data
        self.leftptr:data = None
        self.rightptr:data = None
    
    def __le__(self, _node):
        return (self.userdata <= _node)
    
    def __ge__(self, _node):
        return (self.userdata >= _node)

class binaryTree:
    def __init__(self):
        self.root = None

    def addNode(self, _data):
        if self.root==None:
            newData = node(_data)
            self.root=newData
        else:
            self.insert(self.root, _data)

    def insert(self, currptr, _data):
        if _data >= currptr.userdata:
            if currptr.rightptr == None:
                currptr.rightptr = node(_data)
                return
            else:
                self.insert(currptr.rightptr, _data)
        else:
            if _data <=currptr.userdata:
                if currptr.leftptr == None:
                    currptr.leftptr =node(_data)
                    return
                else:
                    self.insert(currptr.lefptr, _data)
    
    def inorder_traversal(self):
        result=[]
        self._inorder_traversal(self.root, result)
        return result

    def _inorder_traversal(self, node, result):
        if node:
            self._inorder_traversal(node.leftptr, result)
            result.append(node.userdata)
            self._inorder_traversal(node.rightptr, result)
    
    
def main():
    tree = binaryTree()
    data1= data("ravi", 100)
    data2= data("bango", 232)
    data3= data("werer", 50)
    tree.addNode(data1)
    tree.addNode(data2)
    tree.addNode(data3)

    _list= tree.inorder_traversal()

    print(_list)

    for item in _list:
        print(item)

if __name__ == "__main__":
    main() 