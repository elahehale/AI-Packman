import heapq

from util import PriorityQueue
h =  PriorityQueue()
h.push(12,4)
h.push(11,3)
h.push(44,4)

h.update(13,1)
print(h.heap)
