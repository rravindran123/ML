from dataclasses import dataclass

@dataclass
class dataNode:
    def __init__(self, name, salary):
        self.name: str = name
        self.salary: int = salary

    def __eq__(self, obj):
        if obj is None:
            return False
        return self.name == obj.name

    def __lt__(self, obj):
        if obj is None:
            return False
        return self.name < obj.name

    def __gt__(self, obj):
        if obj is None:
            return False
        return self.name > obj.name
    
    def __str__(self):
        return f"name: {self.name}, salary: {self.salary}"

@dataclass
class node:
    userData: dataNode
    gtptr: 'node' = None
    ltptr: 'node' = None

class binaryTree:
    def __init__(self):
        self.root: node = None

    def findNode(self, user):
        currNode = self.root
        parentNode = None

        if currNode is None:
            print("No elements in the tree")
        else:
            while currNode != None:
                #print(f"comparing {currNode.userData} and {user}")
                if currNode.userData == user:
                    print(f"Found Node {currNode.userData}")
                    break
                elif currNode.userData > user:
                    parentNode = currNode
                    currNode = currNode.ltptr
                elif currNode.userData< user:
                    parentNode = currNode
                    currNode =currNode.gtptr

        if currNode == None:
            print("Node coudldnt be found ")

        return currNode, parentNode

    def findMinNode(self, root:node):
        returnRef = root
        while returnRef != None:
            if returnRef.ltptr != None:
                returnRef = returnRef.ltptr
        return returnRef

    def deleteUser(self, user):
        if self.root== None:
            print("the tree is empty")
        else:
            self.delete(self.root, user)

    def delete(self, root,  user):
        if root is None:
            return root
        if user > root.userData:
            root.gtptr = self.delete(root.gtptr, user)
        elif user < root.userData:
            root.ltptr =self.delete(root.ltptr, user)
        elif user == root.userData:
            if root.gtptr == None and root.ltptr ==None:
                return None
            elif root.gtptr == None:
                return root.ltptr
            elif root.ltptr == None:
                return root.gtptr
            noderef = self.findMinNode(root.ltptr)
            root.userData = noderef.userData
            self.delete(self, root.ltptr, noderef.userData)

        return root
        
        

    def addNode(self, userdata: dataNode):
        # If the root is None, initialize it with a new node
        if self.root is None:
            self.root = node(userdata)
            return

        # Traverse the tree to find the appropriate location
        currentNode = self.root
        while True:
            # If userdata is greater, go to the right subtree
            if userdata > currentNode.userData:
                if currentNode.gtptr is None:
                    currentNode.gtptr = node(userdata)  # Assign new node
                    break
                else:
                    currentNode = currentNode.gtptr
            # If userdata is smaller, go to the left subtree
            elif userdata < currentNode.userData:
                if currentNode.ltptr is None:
                    currentNode.ltptr = node(userdata)  # Assign new node
                    break
                else:
                    currentNode = currentNode.ltptr
            else:
                # Duplicate dataNode values are not allowed
                print("Duplicate data. Node not added.")
                break


    def inorderTraverseTree(self):
        if self.root is not None:
            userref = self._inorder(self.root)
        else:
            print("Tree is empty")
        # Start traversal from root if no pointer is provided

    def _inorder(self, node):
        # Terminate recursion if current node is None
        if node is not None:
            noderef= self._inorder(node.ltptr)
            print(f"Node: {node.userData.name}, Salary: {node.userData.salary}")
            noderef= self._inorder(node.gtptr)        

def start():
    # Create data nodes
    data1 = dataNode('a', 10000)
    data2 = dataNode('b', 5000)
    data3 = dataNode('c', 3000)

    # Create a binary tree and add nodes
    tree = binaryTree()
    tree.addNode(data1)
    tree.addNode(data2)
    tree.addNode(data3)

    try:
        file = open("userdata.txt", 'r')
        for line in file:
            name, salary = line.split(" ")
            salary = int(salary)
            userdata = dataNode(name, salary)
            tree.addNode(userdata)
    except:
        print("error in opening the file ")

    # Perform in-order traversal
    print("In-order traversal of the binary tree:")
    tree.inorderTraverseTree()

    data4= dataNode('yuy', 57676)

    print(f"finding {data4}")

    tree.findNode(data4)

    print("Deleting data4")
    tree.deleteUser(data4)

    print("tree after deletion")
    tree.inorderTraverseTree()



if __name__ == "__main__":
    start()
