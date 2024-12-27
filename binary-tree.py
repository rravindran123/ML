import os
from dataclasses import dataclass

class data:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
    
    def __eq__(self, value):
        return (self.name == value.name and self.salary== value.salary)
    
    def __str__(self):
        return f"name: {self.name} and salar: {self.salary}"
    
class node:
    def __init__(self):
        self.userdata: data = data
        self.leftptr:data = None
        self.rightptr:data = None

class binaryTree:
    def __init__(self):
        self.root = None

    def addNode(self, _data):
        if self.root==None:
            newData = data(_data)

