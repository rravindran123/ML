
import string
import sys
sys.setrecursionlimit(10000)


class State:
    def __init__(self):
        self.state={}
        self.state["B1"]=[]
        self.state["B2"]=[]   

    def setState(self, actors:list, state:str):
        for actor in actors:
            self.state[state].append(actors)
    
    def __eq__(self, value):
        if isinstance(value, State):
            return self.state == value.state
        else:
            return False

class SearchAlgotithm:
    def __init__(self, actors=None, state=None, action=None, cost=None):
        self.actors = actors if actors is not None else []
        self.startState = state if state is not None else []
        self.action = action if action is not None else []
        self.cost = cost if cost is not None else {}
    
    def initState(self, _state):
        self.startState = _state

    def _enumerate(self):
        pass
        # enumerate all possible state and actions

class TransportationProblem():
    def __init__(self, N):
        # N is the number of stops
        self.N = N

    def startState(self):
        return 1
    
    def endState(self, state):
        return state==self.N
    
    def succAndCost(self, state):
        result = []
        if state+1 <= self.N:
            result.append(('walk', state+1,1 ))
        if state*2 <= self.N:
            result.append(('tram', state*2, 2))
        return result

def backtrackSearch(problem):
    #keep track of the best solution so far
    bestSolution = {
        'cost': float('inf'),
        'history': None
    }
    def recurse(state, history, totalCost):
        #At State, having undergone history, with total cost
        #Explore the subtree from this state
        if problem.endState(state):
            #Update the best solution so far
            #Todo
            if totalCost < bestSolution['cost']:
                bestSolution['cost'] = totalCost
                bestSolution['history'] = history
            return
        for action, newState, cost in problem.succAndCost(state):
            recurse(newState, history+[(action, newState, cost)], totalCost+cost)
    recurse(problem.startState(), [], 0)
    return (bestSolution['cost'], bestSolution['history'])

def dynamicProgramming(problem):
    bestSolution = {
        'cost': float('inf'),
        'history': None
    }
    cache ={} #state -> futurecost
    def futureCost(state, history, totalCost):
        if problem.endState(state):
            if totalCost < bestSolution['cost']:
                bestSolution['cost'] = totalCost
                bestSolution['history'] = history
            return 0
        if state in cache:
            return cache[state]
        result = min(cost + futureCost(newState, history+[(action, newState, cost)], totalCost+cost ) \
                 for action , newState, cost in problem.succAndCost(state))
        cache[state]=result
        return result
    futureCost(problem.startState(), [],0)
    return (bestSolution['cost'], bestSolution['history'])



def printSolution(solution):
    cost, history = solution
    print(f"Cost: {cost}")
    if history is not None:
        for step in history:
            print(step)
    else:
        print("No solution found")

def main():
    problem = TransportationProblem(10)
    print(problem.succAndCost(9))
   # printSolution(backtrackSearch(problem))
    printSolution(dynamicProgramming(problem))

# def main():
#     actors = ["F", "G", "W", "C"]
#     actions = ["UP", "DOWN"]
#     state = State()
#     state.setState(actors, state )

if __name__ == "__main__":
    main()