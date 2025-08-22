from bisect import bisect_left
from bisect import bisect_right

class telemetry:
    def __init__(self):
        self.systemdata = dict()


    def sortdata(self, metric_name):
        self.systemdata[metric_name].sort(key=lambda tup:tup[0])

    def record(self,metric_name:str, value:float, timestamp:int) -> None:

        if metric_name not in self.systemdata:
            self.systemdata[metric_name]=[(timestamp, value)]
        else:
            self.systemdata[metric_name].append((timestamp, value))
            self.sortdata(metric_name)


    def query(self, metric_name:str, start_time:int, end_time:int, agg:str) -> float:

        if metric_name in self.systemdata:
            data = self.systemdata[metric_name]
            timestamps = [t[0] for t in data]
            dataitems = [t[1] for t in data]
            lo = bisect_left(timestamps, start_time)
            hi = bisect_right(timestamps, end_time)

            if agg =="avg":
                totalsum = sum(dataitems[lo:hi])
                return totalsum/len(dataitems[lo:hi])
            
            if agg =="sum":
                return sum(dataitems)
            
            if agg=="min":
                return min(dataitems)
            
            if agg=="count":
                return len(dataitems)
            
        return None
        
        
metrics = telemetry()

metrics.record("cpu", 0.5, 100)
metrics.record("cpu", 0.7, 105)
metrics.record("cpu", 0.6, 103)

print(metrics.query("cpu", 100, 105, "avg"))
