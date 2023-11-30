'''
1. generate random cities selection
2. find routelength for that generated cities sequence
3. find all neigbours
4. find best neighbour
'''

import random
# 2) select random cities & append in solution list
def randomSolution(tsp):
    cities = list(range(len(tsp)))  #cities list e.g. [0,1,2,3]
    solution = []

    for i in range(len(tsp)):
        randomCity = cities[random.randint(0, len(cities) - 1)] #at first puck any random city
        solution.append(randomCity) 
        cities.remove(randomCity) # we cant consider any city that is already considered

    return solution

#3) find the route length
def routeLength(tsp, solution):
    routeLength = 0
    for i in range(len(solution)):
        routeLength += tsp[solution[i - 1]][solution[i]]
    return routeLength

#4) get all neighbours
def getNeighbours(solution):
    neighbours = [] # list of all neighbours
#loop to check all possibilities of neighbors possible
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)): 
            neighbour = solution.copy() # 1 set of cities is already kept
            #swapping done of neighbours
            neighbour[i] = solution[j]
            neighbour[j] = solution[i]
            neighbours.append(neighbour)
    return neighbours
#5) get the best out of all neighbours
def getBestNeighbour(tsp, neighbours):
    bestRouteLength = routeLength(tsp, neighbours[0])
#initially we consider first set of neighbors set to be the ideal one
    bestNeighbour = neighbours[0]
    for neighbour in neighbours: #iterate in all neighbours
        currentRouteLength = routeLength(tsp, neighbour)
        if currentRouteLength < bestRouteLength:
            bestRouteLength = currentRouteLength
            bestNeighbour = neighbour
    return bestNeighbour, bestRouteLength


def hillClimbing(tsp):
    currentSolution = randomSolution(tsp) #calls for random cities selection
    currentRouteLength = routeLength(tsp, currentSolution) #calls for routelength of cities recieved
    neighbours = getNeighbours(currentSolution) # get neighbours where to move next
    bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(tsp, neighbours)

    while bestNeighbourRouteLength < currentRouteLength: # check if next neighbour is better than current node
        currentSolution = bestNeighbour # update till we find better neighbour
        currentRouteLength = bestNeighbourRouteLength
        neighbours = getNeighbours(currentSolution)
        bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(tsp, neighbours)

    return currentSolution, currentRouteLength

#1) start from deciding the route length

tsp = [
        [0, 400, 500, 300],
        [400, 0, 300, 500],
        [500, 300, 0, 400],
        [300, 500, 400, 0]
    ]

print(hillClimbing(tsp))


