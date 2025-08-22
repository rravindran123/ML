

def findSumIntegerIndices(_list:list, target:int):
    #print("here")
    numDict ={}
    for i in range(len(_list)):
        numDict[_list[i]]=i

    found = False
    # for i in range(len(_list)):
    #     for j in range(len(_list)):
    #        # print(f"checking {_list[i]} and {_list[j]}")
    #         if _list[i] + _list[j] == target:
    #             print(f"index {i}, index{j}")
    #             found =True
    #         else:
    #             continue
    # if found == False:
    #     print("No matching numbers")

    for i in range(len(_list)):
        firstIndex =i
        diff = target - _list[i]
        # if diff < 0:
        #     continue
        if diff in numDict:
            secIndex = numDict[diff]
            found= True
            break
        else:
            continue

    if found is True:
        print(f" {firstIndex}, {secIndex}")


def findTargetSumIndicesOptim(numList:list, target:int):
    numHash = {}
    found = False
    for i in range(len(numList)):
        complement = target - numList[i]
        if complement in numHash:
            found = True
            index1 = i
            index2 = numHash[complement]
            break
        else:
            numHash[numList[i]] =i
    
    if found:
        print(f"{index1}, {index2}")

def matchingBrackets(inputString :str):
    parsedList = []
    inorder= False
    for char in inputString:
        if char in ['{', '(', '[' ]:
            parsedList.append(char)
        elif char in ['}', ')',']' ]:
            if len(parsedList)==0:
                print("No matchign found - not valid")
                break
            topChar = parsedList.pop()
            if topChar == '{' and char == '}' or topChar == '[' and char == ']' or topChar == '(' and char == ')':
                print("matched")
                continue
            else:
                print("Not a valid string")
                break
    
    return            

def threeNumSum(nlist:list):
    #sort the list
    nlist.sort()
    print(f"sorted list :{nlist}")
    indexList=[]

    print(nlist)

    for i in range(len(nlist)-2):
        if i >0 and nlist[i]==nlist[i-1]:
            continue
        
        leftIndex = i+1
        rightIndex = len(nlist)-1

        while leftIndex < rightIndex:

            currentSum = nlist[i] + nlist[leftIndex] + nlist[rightIndex]

            if currentSum ==0 :
                indexList.append((i, leftIndex, rightIndex))

                leftIndex +=1
                while leftIndex < rightIndex and nlist[leftIndex ]== nlist[leftIndex+1]:
                    leftIndex +=1

                rightIndex -=1
                while rightIndex > leftIndex and nlist[rightIndex ]== nlist[rightIndex-1]:
                    rightIndex -=1

            if currentSum <0:
                leftIndex+=1
            else:
                rightIndex-=1
    return indexList

from dataclasses import dataclass
import time
@dataclass
class Wordgroups:
    signature: str
    anagrams: list

def Getsignature(word:str, charcount:bool=False)->tuple:
        if charcount:
            counts =[0]*26
            for char in word:
                counts[ord(char)- ord('a')]+=1
            return tuple(counts)
        else:
            return ''.join(sorted(word))


def FindAnagrams(strlist: list):
    #sort by the len of the words
    # wordlistlensorted= sorted(strlist, key= lambda x : len(x))
    # print(wordlistlensorted)
    start = time.time()
    anagramdict ={}
    alphabetsortlist =[]
    for word in strlist:
        #sig = ''.join(sorted(word))
        sig = Getsignature(word, True)
        if sig not in anagramdict:
            anagramdict[sig] = Wordgroups(sig,[])
        anagramdict[sig].anagrams.append(word)

    returnList=[]
    for words in anagramdict.values():
        returnList.append(words.anagrams)

    end= time.time()

    print(f"runtime :{end-start}")
    
    return returnList

def main():
    #intList = [100, 2, 4, 6, 7, 23, 15]
    #print("Testing")
    #findTargetSumIndicesOptim(intList, 115)
    # matchingBrackets("{[]}()")
    # nums = [-1, -1, 0, 1, 2, -1, -4, -3]
    # print(threeNumSum(nums))
    strs = ["ea","ae","tan", "lame" , "ate","eat","tae", "nat","bat", "emal"]
    result= FindAnagrams(strs)
    print(result)

if __name__ == "__main__":
    main()
        

