#!usr/bin/python 3.7
#-*-coding:utf-8-*-

import subprocess 
import psutil
import matplotlib.pyplot as plt
import time

cmd = "python memory_test.py"
process = subprocess.Popen(cmd.split(" "))

pid = process.pid
print("process id: ", pid)

def get_memory_list():
    process = psutil.Process(pid)
    memory_list = []
    while process_running(process):
        try:
            memo = process.memory_info().rss / 1024 / 1024 #MB
        except:
            break
        memory_list.append(memo)
        time.sleep(1)
    return memory_list

def process_running(process):
    try:
        memo = process.memory_info().rss / 1024 / 1024
        return True 
    except:
        return False

def plot():
    start = time.time()
    memory_list = get_memory_list()
    end = time.time()
    print("Time spent to run {}s".format(round(end-start, 2)))
    plt.plot([x for x in range(len(memory_list))], memory_list)
    plt.xlabel("record point")
    plt.ylabel("memory (MB)")
    plt.show()

if __name__ == "__main__":
    plot()
