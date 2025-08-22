from typing import Optional
from collections import deque

class Node:
    def __init__(self, _nodeid:int, _label:Optional[str]=None):
        self.nodeid = _nodeid
        self.label:Optional[str] = _label
        self.edgelist:dict(int, Edge) = {}


    def __repr__(self):
        return f"node:{self.nodeid} label:{self.label}"
    
class Edge:
    def __init__(self, _neighborid, weight:int):
        self.nodeid = _neighborid
        self.weight = weight

    def __repr__(self):
        return f"edge {self.nodeid} weight {self.weight}"

class Graph:
    def __init__(self, undirected = False):
        self.graph: dict(int, node)={}

    def add_edge(self,node1, node2, weight):
        if node1 not in self.graph:
            self.graph[node1] = Node(node1)

        if node2 not in self.graph:
            self.graph[node2] = Node(node2)

        # add the edge to Node1
        if node2 not in self.graph[node1].edgelist:
            self.graph[node1].edgelist[node2] =Edge(node2, weight)


    def __repr__(self):
        ret_str = f"Printing Graph \n"
        for nodeid, node in self.graph.items():
            ret_str = ret_str + f"{node}"
            for edgeid, edge in node.edgelist.items():
                ret_str = ret_str + f" -> {edge}"
            ret_str = ret_str + "\n"
        return ret_str

def dfs(_graph, src):
    if src not in _graph.graph:
        print(f"node {src} not in the graph")
        return

    stack = deque()
    stack.append(src)
 #   visited = deque()
    parent = dict()

    while stack:
        curr_node = stack.pop()
    #    visited.append(curr_node)

        if curr_node ==src:
            parent[src]=None
            
        for key, edge in _graph.graph[curr_node].edgelist.items():
            if edge.nodeid not in parent:
                stack.append(edge.nodeid)
                #update the parent
                parent[edge.nodeid]=curr_node
                
    #printing the tree traversal
    print("Dfs traversal..\n")
    # while len(visited) >0:
    #     curr_node = visited.popleft()
    #     if parent[curr_node] != None:
    #         print(f"edge : {parent[curr_node]}->{curr_node}")

    for  node, _parent in parent.items():
        if _parent != None:
            print(f"edge : {parent[node]}->{node}")
    
    return

def bfs(_graph, src):
    if src not in _graph.graph:
        print(f"node {src} not in the graph")
        return

    queue = deque()
    queue.append(src)
  #  visited = deque()
    parent = dict()
    parent[src]=None
    dist= dict()
    dist[src]=0

    while queue:
        curr_node = queue.popleft()
      #  visited.append(curr_node)

        for key, edge in _graph.graph[curr_node].edgelist.items():
            if edge.nodeid not in parent: # the node should be checkd in the parent for bfs, else the shortest path will be missed
                queue.append(edge.nodeid)
                #update the parent
                parent[edge.nodeid]=curr_node
                dist[edge.nodeid] = dist[curr_node]+1
                
    #printing the tree traversal
    print("Bfs traversal..\n")
    # while len(visited) >0:
    #     curr_node = visited.popleft()
    #     if parent[curr_node] != None:
    #         print(f"edge : {parent[curr_node]}->{curr_node}")

    for  node, _parent in parent.items():
        if _parent != None:
            print(f"edge : {parent[node]}->{node}, shortest distance {dist[node]}")
    
    return
import math

def dijkstras (_graph, src, dest):

    shortest_dist:dict[int, int] ={}
    parent:dict[int, Optional[int]|None]={}
    child: dict[int, Optional[int]|None ]={}
    
    if src not in _graph.graph or dest not in _graph.graph:
        print("src or dest not found in the graph")
        return
    
    parent[src]= None
    for node_id in _graph.graph:
        if node_id == src:
            shortest_dist[node_id]=0
        else:
            shortest_dist[node_id]=math.inf
        
    unlabelled_node = {src: 0}
    parent[src]=None

    while unlabelled_node:

        curr_node = min(unlabelled_node, key=unlabelled_node.get)
        del unlabelled_node[curr_node]

        if curr_node == dest:
            print(f"Labelled the destination, the cost is {shortest_dist[curr_node]}")
            child[curr_node]=None
            while curr_node != None:
                child[parent[curr_node]] = curr_node
                curr_node = parent[curr_node]
                
            break

        edge_list = _graph.graph[curr_node].edgelist
        
        for neighbor, edge_info in edge_list.items():
            if shortest_dist[curr_node] + edge_info.weight < shortest_dist[neighbor]:
                shortest_dist[neighbor] = shortest_dist[curr_node] + edge_info.weight
                unlabelled_node[neighbor]=shortest_dist[neighbor]
                parent[neighbor] = curr_node
                print(f"updated {neighbor} with cost {shortest_dist[neighbor]}")
        
        #sort the unlabelled list 
    #print path
    curr_node = src
    cost =0
    print("Printing path..\n")
    while curr_node != None:
        if child[curr_node] != None:
           # print(f"Edge {parent[curr_node]} -> {curr_node}")
            print(f"Edge {curr_node} -> {child[curr_node]} : weight: {_graph.graph[curr_node].edgelist[child[curr_node]].weight}")
            cost += _graph.graph[curr_node].edgelist[child[curr_node]].weight
        curr_node = child[curr_node]
    print(f"Path cost : {cost}")  

    return

g= Graph()
g.add_edge(1,2,2)
g.add_edge(1,3,5)
g.add_edge(2,5,19)
g.add_edge(2,4,6)
g.add_edge(3,4,10)
g.add_edge(3,5,10)
g.add_edge(4,6,10)
g.add_edge(5,6,10)
print(g)

#calculate a dfs from a source node
dfs(g,1)
bfs(g,1)

dijkstras(g, 1,6)