
class stack:
    def __init__(self):
        self.objectList = list()
        self.minList = list()

    def push(self, value:int):
        self.objectList.insert(0,value) 

        if not self.minList or value<=self.minList[-1]:
            self.minList.insert(0,value)
        else:
            self.minList.insert(0,self.minList[-1])

    def pop(self)->int:
        val= self.objectList.pop(0)
        self.minList.pop(0)
        return val

    def minValue(self):
        return self.minList[0]
    
    def __str__(self):
        print(f"Min list :{self.minList}")
        items=[str(self.objectList[i]) for i in range(len(self.objectList))]
        return ", ".join(items)



def main():
    objList = stack()

    objList.push(-80)
    objList.push(1)
    objList.push(-2)
    objList.push(100)
    objList.push(-200)

    print(objList)
    print("Minimum vale", objList.minValue())
    print("popping",objList.pop())
    print("popping",objList.pop())
    print("Minimum vale", objList.minValue())

if __name__=="__main__":
    main()