import heapq


class PriorityQueue:
    '''
    Queue object to maintain the list of nodes to visit and update
    '''
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def insert(self, vertex, priority_keys):
        '''
        to insert elem into queue using vertex and its keys
        '''
        heapq.heappush(self.elements, (priority_keys, vertex))

    def pop_smallest(self):
        '''
        to pop the elem with smallest key
        '''
        return heapq.heappop(self.elements)

    def top_key(self):
        '''
        to return the smallest k0 val
        '''
        # print(self.elements)
        return heapq.nsmallest(1, self.elements)[0][0]

    def delete(self, node):
        self.elements = [e for e in self.elements if e[1] != node]
        heapq.heapify(self.elements)

    def __iter__(self):
        for key, node in self.elements:
            yield node