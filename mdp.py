import random
import matplotlib.pyplot as plt
import os

actions = ['quit', 'play']
state = ['start', 'stay', 'end']
dice = ['1', '2', '3', '4', '5', '6']
distribution={}
epoch =10000

def diceGame(stayProb=0.5):
    for i  in range(epoch):
        currentState = 'start'
        totalRewards =0
        while currentState != 'end':

            #newChoice = random.choice(actions)
            randVal = random.random()

            if randVal < stayProb:
                newChoice = 'play'
            else:
                newChoice = 'quit'

            if newChoice =='quit':
                currentState = 'end'
                totalRewards += 10
                continue
            elif currentState == 'start' and newChoice == 'play':
                currentState = 'stay'
                totalRewards += 4
                diceroll = random.choice(dice)
                # or use radom.randint(1,6)
                if diceroll == '1' or diceroll == '2':
                    currentState = 'end'
                else:
                    currentState = 'stay'
        if totalRewards in distribution:
            distribution[totalRewards] +=1
        else:
            distribution[totalRewards]=1

    print(f"Final rewards {totalRewards}")
    expectedRewads =0
    sortedDist = {k:distribution[k]/epoch for k in sorted(distribution)}
    for key, value in sortedDist.items():
        print(f" Reward:{key}, distribution:{sortedDist[key]}")
        expectedRewads += key*sortedDist[key]
    print(f"Expected Reward {expectedRewads}")

    keys = list(sortedDist.keys())
    values = list(sortedDist.values())

    # Create a bar plot
    plt.bar(keys, values)
    plt.title('Keys vs. Values (Bar Chart)')
    plt.xlabel('Keys')
    plt.ylabel('Values')
    plt.grid(axis='y', linestyle='--')  # Optional grid
    plt.show()



class transportMdp:
    def __init__(self,n:int, failureProb:float=0.5):
        self.N = n
        self.failureProb = failureProb
    
    def isEnd(self, state):
        return state == self.N

    def nextAction(self, state):
        actions=[]
        if state + 1 <=self.N:
            actions.append(('walk'))
        if state*2 <= self.N:
            actions.append(('tram'))

        return actions

    def succActionRewards(self, state, action):
        result=[]
        if action =='walk':
            result.append((state+1, 1.0, -1))
        elif action =='tram':
                result.append((state*2, 1-self.failureProb, -2))
                result.append((state, self.failureProb, -2))
        return result
    def discount(self):
        return 1.
    def states(self):
        return range(1, self.N+1)

def valueIteration(mdp):
    V={} # maps state to the value of the state
    for state in mdp.states():
        V[state]=0

    def Q(state, action):
        return sum(prob*(reward + mdp.discount() * V[newState])\
                for newState, prob, reward in mdp.succActionRewards(state, action))  
    while True:
        newV ={}
        for state in mdp.states():
            if mdp.isEnd(state):
                newV[state]=0
            else:
                newV[state]= max(Q(state, action) \
                                for action in mdp.nextAction(state))
                # qvalue=[]
                # for action in mdp.nextAction(state):
                #     #print(f"action : {action[0]}, reward: {action[1]}")
                #     #print(f"Qvalue : {Q(state,action[0])}")
                #     qvalue.append(Q(state,action))
                # newV[state] = max(qvalue)

                print(f"State: {state}, Value: {newV[state]}")
        #check for convergence
        if max(abs(V[state]- newV[state]) for state in mdp.states()) < 1e-10:
            break
        V = newV

        #read out the policy
        pi = {}
        for state in mdp.states():
            if mdp.isEnd(state):
                pi[state]='none'
            else:
                pi[state]= max((Q(state, action), action) for action in mdp.nextAction(state))[1]
                print({pi[state]})
        #print the progress
        os.system('clear')
        for state in mdp.states():
            print(f"{state}, {V[state]}, {pi[state]}")
        input()


class Halvinggame:
    def __init__(self, N):
        self.N = N
    
    def startstate(self):
        return (+1, self.N)
    
    def endstate(self, state):
        player, number = state
        return number ==0
    
    def utility(self, state):
        player, number = state
        assert number ==0
        return player*float('inf')
    
    def actions(self, state):
        return ['-', '/'] 
    
    def player(self , state):
        player, number = state
        return player
    
    def succ(self, state, action):
        player, number = state
        if action == '-':
            return (-player, number-1)
        elif action == '/':
            return (-player, number//2)

def Humanpolicy(game, state):
    while True:
        action = input('Input action: ')
        if action in game.actions(state):
            return action

def playgame():
    policies = {+1: Humanpolicy, -1:Humanpolicy}
    game = Halvinggame(15)
    state = game.startstate()

    while not game.endstate(state):
        print('='*10, state)
        player = game.player(state)
        policy = policies[player]
        action = policy(game, state)
        state = game.succ(state, action)
    
    print(f"utility {game.utility(state)}")


def main():
    #diceGame(0.8)
    #transportProb = transportMdp(10,0.1)
    # print(transportProb.nextAction(5))
    # print(transportProb.succActionRewards(5, 'tram'))
    
    #valueIteration(transportProb)
    playgame()


if __name__ == "__main__":
    main()