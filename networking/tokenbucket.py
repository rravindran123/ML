import time
import numpy as np
#print(time.time())

from dataclasses import dataclass

#@dataclass
class request:
    def __init__(self,requestid, currtime, new_tokens):
        self.requestid=requestid
        self.lastrequest=currtime
        self.tokens = new_tokens

    def update(self, reqid, currtime, new_tokens):
        self.requestid=reqid
        self.lastrequest=currtime
        self.tokens = new_tokens

    def __repr__(self):
        return f" id: {self.requestid}, lastreq: {self.lastrequest}, tokens:{self.tokens}"

class client_token_bucket:
    def __init__(self, rate:int, burst:int):
        self.tokenrate= rate
        self.tokenburst = burst
        self.reqList = dict() # key clientid , value metadata about the client

    def checkRequestRate(self, clientid, reqid, currtime):
        #currtime = time.time()
        if clientid not in self.reqList:
            #self.reqList[clientid]= []
            num_tokens = self.tokenburst
            self.reqList[clientid]=request(reqid, currtime, num_tokens)
            print(f"Client {clientid} new request allowed, tokens { self.reqList[clientid].tokens:.2f}")
        else:
            # replenish the token bucket
            requeststate = self.reqList[clientid]
            #print(f"Client {clientid} request allowed, tokens {requeststate.tokens}")
            new_tokens = self.tokenrate * (currtime - requeststate.lastrequest)
            allowed_tokens = min(self.tokenburst, new_tokens+requeststate.tokens)
            self.reqList[clientid].update(reqid, currtime, allowed_tokens)
            #print(f"Client {clientid} {new_tokens} {allowed_tokens}")

            if requeststate.tokens >= 1:
                print(f"Client {clientid} request allowed, tokens {requeststate.tokens}")
                requeststate.tokens -=1
            else:
                print(f"Client {clientid} request not allowed tokens: {requeststate.tokens:.2f}")
    

class rate_limiter:
    def __init__(self):
        self.client_db: dict[str, dict]={}
    
    def register_client(self, client_id:str, rate:float, burst:int) -> None:
        if client_id in self.client_db:
            print("Client already registered")
            return
        # add the new client
        self.client_db[client_id]={'rate':rate, 'burst':burst, 'last_checked':0.0, 'tokens':burst}
        return
    
    def allow_request(self, client_id, timestamp:float)->bool:
        
        if client_id not in self.client_db:
            print("Client not registered")
            return

        #update the tokens for the client
        last_event = self.client_db[client_id]['last_checked']
        
        # add the tokens to the client
        total_tokens = (timestamp - last_event)*self.client_db[client_id]['rate'] + self.client_db[client_id]['tokens']
        self.client_db[client_id]['tokens'] = min(total_tokens , self.client_db[client_id]['burst'] )

        print(f"Tokens for {client_id} {self.client_db[client_id]['tokens']}")
        self.client_db[client_id]['last_checked']= timestamp

        if self.client_db[client_id]['tokens'] >0 :
            print(f"Allowing request for client {client_id}")
            self.client_db[client_id]['tokens']-=1
            return True
        else:
            print(f"Denying request for client {client_id}")
            return False
        


def run_tests():
    rl = rate_limiter()

    # Register a client with rate = 1 token/sec and burst = 5 tokens
    rl.register_client("clientA", rate=1.0, burst=5)

    results = []

    # t = 0: clientA can use up to 5 tokens
    t = 0.0
    for _ in range(5):
        results.append(rl.allow_request("clientA", t))  # Should be True

    results.append(rl.allow_request("clientA", t))  # Should be False (burst exhausted)

    # t = 2: 2 tokens refilled
    t = 2.0
    results.append(rl.allow_request("clientA", t))  # Should be True
    results.append(rl.allow_request("clientA", t))  # Should be True
    results.append(rl.allow_request("clientA", t))  # Should be False (only 2 tokens refilled)

    # t = 7: 5 seconds passed since t=2, refill up to burst cap (5)
    t = 7.0
    for _ in range(5):
        results.append(rl.allow_request("clientA", t))  # Should be True

    results.append(rl.allow_request("clientA", t))  # Should be False again

    expected = [
        True, True, True, True, True,  # Initial burst
        False,                         # Exceeds burst
        True, True, False,            # 2 tokens refilled
        True, True, True, True, True,  # Full refill
        False                          # Exceeds again
    ]

    for i, (r, e) in enumerate(zip(results, expected)):
        assert r == e, f"Test {i+1} failed: got {r}, expected {e}"

    print("✅ All tests passed!")

run_tests()    



    

# currtime =0 
# simtime = 10
# meanrate = 30
# meaninterarrivaltime = 1/meanrate
# maxrequests = 1000
# count =0

# usertokenbucket = client_token_bucket(10, 10)
# clientlist = ["1", "2", "3"]

# while count < maxrequests:
#     currtime+= np.random.exponential(meaninterarrivaltime)
#     client = np.random.choice(clientlist)
#     reqid = client + str(time.time())
#     usertokenbucket.checkRequestRate(client, reqid, currtime)
#     count+=1