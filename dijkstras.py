from dataclasses import dataclass

@dataclass
class Node:
    nodeId: int
    cost : int
    edgeList: list
    previousNode: int

@dataclass
class Edge:
    adjacent_node : Node
    edgeWeight : int


class directedGraph:
    def __init__(self, nodeSize: int =0, edgeSize: int=0):
        self.nodes = nodeSize
        self.edges = edgeSize
        self.graph = dict()
        for i in range(self.nodes):
            edgelist= list()
            self.graph[i]= Node(i, 99999, [], -1)

    def printGraph(self):
        print(f"\n# of nodes {self.nodes}, # of Edge {self.edges}")
        for  i in range(self.nodes):
            node = self.graph[i]
            for edge in node.edgeList:
                print(f"Edge {node.nodeId} -> {edge.adjacent_node}, weight: {edge.edgeWeight}")
    
    def addEdge(self, sourceNode: int , destNode: int , weight:int ):
        #get the edge list
        assert sourceNode < self.nodes and destNode < self.nodes
        node = self.graph[sourceNode]
        edgeList = node.edgeList
        edge_exist = False
        for edge in edgeList:
            if edge.adjacent_node == destNode:
                edge_exist = True
                print("Edge already exists")
        if edge_exist == False:
            newEdge = Edge(destNode, weight)
            edgeList.append(newEdge)

    def calculateShortestPath(self, sourceNode, destNode):
        tempLabeled=[]
        permantentlyLabeled=[]
        for i in range(self.nodes):
            node = self.graph[i]
            if node.nodeId == sourceNode:
                node.cost=0
                tempLabeled.append(node)
            else:
                tempLabeled.append(node)
        
        tempLabeled.sort(key=lambda node: node.cost)

        while len(tempLabeled) > 0:
            # get the node with the least cost
            leastCostNode = tempLabeled.pop(0)
            print(f"least cost node id {leastCostNode.nodeId}, cost {leastCostNode.cost}")

            if leastCostNode.nodeId == destNode:
                print(f"Destination node {leastCostNode.nodeId} is permanently labeled with cost {leastCostNode.cost}")
                break
            #update the cost of the adjacent nodes if the new cost update is lower than the current cost
            edgeList = leastCostNode.edgeList

            for e in edgeList:
                adjacentNode = self.graph[e.adjacent_node]
                if leastCostNode.cost + e.edgeWeight < adjacentNode.cost:
                    # found a path with lower cost
                    prevCost = adjacentNode.cost
                    adjacentNode.cost = leastCostNode.cost + e.edgeWeight
                    adjacentNode.previousNode = leastCostNode.nodeId
                    print(f"updated {adjacentNode.nodeId} with prev cost {prevCost} with new cost {adjacentNode.cost}")
            
            #resort the temporarily labeled nodes
            tempLabeled.sort(key=lambda node:node.cost)
        
        #reverse trace the path
        print("This is the least cost path")
        currentNode = self.graph[destNode]
        pathcost =0
        while currentNode.previousNode !=-1 :
            prevNode = self.graph[currentNode.nodeId].previousNode
            for e in self.graph[prevNode].edgeList:
                if e.adjacent_node == currentNode.nodeId:
                    print(f" Edge {prevNode} -> {currentNode.nodeId} weight : {e.edgeWeight}")
                    pathcost+= e.edgeWeight
                    currentNode = self.graph[prevNode]
                    break
        
        print(f"Total path cost :{pathcost}")


#initialize graph reading a file
def initGraph(fileName):
    try:
        with  open(fileName, 'r') as file:
            nodes_edges = file.readline().strip()
            nodes, edges = map(int, nodes_edges.split(","))
            print(f"nodes : {nodes}, edges: {edges}")
            graph_instance = directedGraph(nodes, edges)

            for line in file:
                edge_line = line.strip()
                node1, node2, edge_weight = map(int, edge_line.split(","))
                print(f"node1 : {node1}, node2 : {node2} weight: {edge_weight}")
                graph_instance.addEdge(node1, node2, edge_weight)

            graph_instance.printGraph()
            return graph_instance
    except  FileNotFoundError:
        print("The file was not found, Please check the file path")
    except  IOError:
        print("A Error occurent openign the file ")

    # read the first line which has the #nodes and # of edges
    #nodes, edges = (file.read()).split()

    
graph= initGraph("graph.txt")
graph.calculateShortestPath(0, 4)
