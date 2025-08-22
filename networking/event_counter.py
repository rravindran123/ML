import time, bisect

class event_counter:
    def __init__(self):
        self.eventlist = list()

    def record(self, timestamp:int):
        self.eventlist.append(timestamp)
    
    def get_event_count(self, window:int):
        event_list_size = len(self.eventlist)
        most_recent = self.eventlist[event_list_size-1]
        low_time = most_recent - window

        index =0
        # for event in self.eventlist:
        #     if event <= low_time:
        #         index +=1
        # gives the first index that is greater than low_time
        index = bisect.bisect_left(self.eventlist, low_time)

        print(f"The ealiest event in timewindow {window} is {self.eventlist[index]}")
        return_list = self.eventlist[index:]

        return return_list
    
# counter = event_counter()
# counter.record(90)
# counter.record(120)
# counter.record(130)
# print(counter.get_event_count(30))  # → 2 (events at 120, 130)
# print(counter.get_event_count(100)) # → 3 (events from 30 seconds ago till now)

class deduplication_service():
    def __init__(self, _time):
        self.dedup_time=_time
        self.reqlist=dict()


    def receive(self, reqid:str, timestamp:int):
        #get the currtime
        currtime = time.time()

        if reqid not in self.reqlist:
            self.reqlist[reqid]=timestamp
            print("Allowed")
            return True
        
        #clean the list with requests that are expired
        for req, _time in self.reqlist.items():
            if currtime-time > self.dedup_time:
                self.reqlist.pop(req)
        
        if reqid not in self.reqlist:
            self.reqlist[reqid]=timestamp
            print("Allowed")
            return True
        else:
            print("Denied")
            return False


deduper = deduplication_service(10)
deduper.receive("abc123", timestamp=100)  # → True (new request)
deduper.receive("abc123", timestamp=105)  # → False (duplicate within T=10 seconds)
deduper.receive("abc123", timestamp=112)  # → True (old request expired, so allowed again)