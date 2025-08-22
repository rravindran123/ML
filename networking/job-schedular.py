import time

class job_schedular:
    def __init__(self, _timeout:int):
        self.timeout = _timeout
        self.tasklist = dict() # task-id -> payload
        self.unassigned_tasks=list()
        self.expired_tasks = list()
        self.inprogess_tasks = dict() # workerid -> (taskid, currtime)
        #self.unfinishedtasks= dict() # taskid -> payload
        

    def enqueue(self, task_id: str, payload: str):
        
        #check and insert the taskid
        if task_id not in self.tasklist:
            self.tasklist[task_id]= payload
            self.unassigned_tasks.append(task_id)
        else:
            print("Task already exists")
    
    def poll(self, worker_id:str):
        currtime = time.time()

        #check if there are any tasks that have not been acknoeledged
        if len(self.inprogess_tasks) >0:
            for task, _worker in self.inprogess_tasks.items():
                if currtime - _worker[1] > self.timeout:
                    print(f"{task} has not been acked, move it to the expired tasks")
                    self.expired_tasks.append(task)
                    self.inprogess_tasks.pop()

        #first assign tasks from the expired task list
        if len(self.expired_tasks) >0 :
            task = self.expired_tasks.pop(0) # get the oldest
            self.inprogess_tasks[worker_id]= (task, currtime)
            print(f"worker {worker_id} assigned expired task {task}")
            return
       
        #assign the oldest task to the
        if len(self.unassigned_tasks) >0:
            task = self.unassigned_tasks.pop(0)
            self.inprogess_tasks[task]=(worker_id, currtime)
            print(f"worker {worker_id} assigned new task {task}")
            return
        else:
            print("no tasks to be assigned")
                
        return
        
    def acknowledge(self, worker_id, task_id):

        currtime = time.time()
        if worker_id in self.inprogess_tasks:
            task, _time = self.inprogess_tasks[worker_id]
        if currtime - _time <= self.timeout:
            print("acknowledging the task is complete")
            self.inprogess_tasks.pop(worker_id)
        else:
            #ack arrived after the timeout
            task, _ = self.inprogess_tasks.pop(worker_id)
            print(f"inserting {task} into the expired task list")
            self.expired_tasks.append(task)
        return
    

q = job_schedular(_timeout=5)
q.enqueue("task1", "process a")
q.enqueue("task2", "process b")

q.poll("worker1") #("task1", "process a")
q.poll("worker2") # ("task2", "process b")
time.sleep(5)
q.poll("worker3") # None


