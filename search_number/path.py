'''
Calculates a path decomposition of G using sage
and then makes the path decomposition "nice". A
"nice" path decomposition has all bags X_i, X_{i+1}
having |symmetric_difference(X_i, X_{i+1})| = 1.
'''

import networkx as nx
from sage.all import *
from sage.graphs.graph_decompositions.vertex_separation import path_decomposition

def checkUsed(vertex, currentPath, graph):
    '''
    Checks whether a vertex is still needed in
    the path decomposition of graph.

    INPUT
    vertex: The vertex which we are checking
    currentPath: A partially completed path decomposition of the 
    form of nicePath in makeNicePathDecomp
    graph: The graph which is being decomposed

    OUTPUT
    used: Boolean on whether the vertex is needed (True if it's not)
    '''
    #Get the neighbors of our current vertex
    neighbors  = graph.neighbors(vertex)

    used = True

    #Go through each neighbor and check whether there is a bag w/ vertex and neighbor
    for neighbor in neighbors:
        included = False
        
        #We want to find a bag where both neighbor and vertex are in the bag
        for bag in currentPath:
            if (vertex in bag) and (neighbor in bag):
                included = True
                break
        
        #If no bag has both in it, then the vertex must appear in future bags
        if not included:
            used = False
            break

    return used

             


def makeNicePathDecomp(G):
    '''
    Creates a "nice" path decomposition of G using
    a path decomposition made by sage.

    INPUT
    G: A networx graph

    OUTPUT
    nicePath: A list of the form [[v1], [v1,v2], ...] in
    which the internal lists are the bags of a "nice" path
    decomposition.
    '''

    #Convert our graph to a sage graph and compute the path decomp
    sageGraph = Graph(G)
    pathDecomp = path_decomposition(sageGraph)[1]
    print pathDecomp
    
    usedVertices = []
    nicePath = [[]]
    i = 0

    #Keep making nice bags until all vertices are used
    while len(usedVertices) != len(pathDecomp):

        prevBag = nicePath[len(nicePath) - 1]
        used = False
        
        #Check to see if any vertices in the previous bag were used up
        for j in range(0, len(prevBag)):
            used = checkUsed(prevBag[j], nicePath, G)

            #Discard the vertex for the next bag if it's used
            if used:
                nextBag = prevBag[:]
		usedVertices.append(nextBag[j])
                del nextBag[j]
                nicePath.append(nextBag)
               
                break

        #If we have found a used vertex we do not need to add any
        if used:
            continue
        
        #Add the next unadded vertex
        nextBag = prevBag[:]
        nextBag.append(pathDecomp[i])
	nicePath.append(nextBag)
        i = i + 1

    return nicePath

G = nx.petersen_graph()
print makeNicePathDecomp(G)
