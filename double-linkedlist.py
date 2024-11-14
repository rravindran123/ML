import asyncio
import time


async def fetch_data(delay, id):
    print("fetching data..id", id)
    await asyncio.sleep(delay)
    print("data fetched")
    return {"data":"Some data", "id":id}

async def main():
    await asyncio.gather(*(fetch_data(i,i) for i in range(5)))
    # print   ("Start the main routine")
    # task_1 = fetch_data(2, 1)
    # task_2 = fetch_data(3,2)
    # result_1 = await task_1 
    # result_2 = await task_2

    # print(f"Received result : {result_1}")
    # print(f"Received result: {result_2}")

 
#     print("End of main coroutine")


start_time = time.time()
asyncio.run(main())
end_time = time.time()

print("Total time: ", (end_time-start_time))
