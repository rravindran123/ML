import random


graph ={
        1 : [2, 3],
        2 : [1, 3],
        3: [1, 2,4],
        4: [2, 3, 5],
        5: [2, 4]
        }

variables = ['x1', 'x2', 'x3']



def graph_coloring_iterative(graph:dict, max_color:int):

    assignment={}
    next_color={}

    def is_valid(node, color):
        for neighbor in  graph[node]:
            if neighbor in assignment and assignment[neighbor]==color:
                return False
        return True

    node_list =[node for node in graph]
    num_nodes= len(graph)

    for node in node_list:
        next_color[node]=1
    
    current_index =0

    while current_index >= 0 and current_index < num_nodes :
        current_node = node_list[current_index]
        color_found = False

        for color_code in range(next_color[current_node], max_color+1):
            if is_valid(current_node, color_code):
                assignment[current_node] = color_code
                next_color[current_node] = color_code+1
                color_found = True
                break
        
        if color_found:
            current_index +=1
            if current_index < num_nodes:
                next_color[node_list[current_index]]=1
        
        else:
            next_color[current_node]=1
            current_index = current_index -1
            if current_index >=0:
                previous_node = node_list[current_index]
                del assignment[previous_node]

    if current_index == num_nodes:
        return assignment
    else:
        return None


def graph_coloring(graph:dict, k:int):
    assignment = {}

    def is_valid(node, color):
        for neighbor in  graph[node]:
            if neighbor in assignment and assignment[neighbor]==color:
                return False
        return True

    def backtrack():
        assigned = True

        if len(assignment) ==len(graph):
            return True
    
        unassigned = [node  for node in graph if node not in assignment]

        # node needs to tbe assigned
        node = unassigned[0]
        for color in range(1, k+1):
            print(f"Trying node: {node} and color: {color}")
            if is_valid(node, color):
                assignment[node]=color
                if backtrack():
                    return True
                print(f"Unassigning node: {node}, color: {color}")
                del assignment[node]
        return False
    
    if backtrack():
        return assignment
    else:
        return None


graph2 = {
        'x1': ['x2'],
        'x2':['x1','x3'],
        'x3': ['x2']
}

def binaryassignment(graph:dict):

    assignment ={}
    solutions=[]

    def is_valid(var1):
        for var2 in graph[var1]:
            if var2 not in assignment:
                continue
            else:
                if assignment[var1] ^ assignment[var2]:
                    continue
                else:
                    return False
        return True
    counter = 0

    def backtrack():
        nonlocal counter
        counter += 1
        print(f"Calling backtrack: {counter}")

        if len(assignment) == len(graph):
            solutions.append(dict(assignment))
            return True

        unassigned = [vars for vars in graph if vars not in assignment]
        next_var = unassigned[0]

        for value in [True, False]:
            assignment[next_var]=value
            if is_valid(next_var):
                if backtrack():
                    return True
                del assignment[next_var]
        return False
    
    if backtrack():
        return assignment
    else:
        return None
        

def test_csat():

    # for node, edges in graph.items():
    #     print(f"Node: {node}, edges: {edges}")

    # assignment = graph_coloring_iterative(graph, 3)

    # if assignment:
    #     for node, color in assignment.items():
    #         print(f"Node {node}, color: {color}")
    # else:
    #     print("Valid assignement not found")

    assignment = binaryassignment(graph2)
    # if assignment:
    #     for node, value in assignment.items():
    #         print(f"Node {node}, assignment: {value}")
    # else:
    #     print("Valid assignement not found")

    print(assignment)

if __name__== "__main__":
    test_csat()



