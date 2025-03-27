
import string


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

problem = TransportationProblem(10)
print(problem.succAndCost(3))

    


# def main():
#     actors = ["F", "G", "W", "C"]
#     actions = ["UP", "DOWN"]
#     state = State()
#     state.setState(actors, state )



# if __name__ == "__main__":
#     main()