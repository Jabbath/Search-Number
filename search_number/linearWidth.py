'''
Calculates the linear width of a graph G using the algorithm given in
[1]: Bodlaender et. al. Computing Small Search Numbers in Linear Time. (1998)
'''
import networkx as nx
from path import makeNicePathDecomp

def isubgraph(i, nicePath, G):
    '''
    Creates the subgraph of G induced by the union of
    nodes in nicePath from 0 to i.

    INPUT
    i: An integer 0 <= i <= len(nicePath) - 1
    nicePath: A nice path decomposition of G
    G: A networkx graph

    OUTPUT
    subgraph: The aformentioned subgraph of G
    '''
    union = set()
    
    #Take the union up to i
    for j in range(0, i + 1):
        union = union.union(nicePath[j])

    subgraph = G.subgraph(list(union))
    return subgraph

def findPendant(edges, p, Gi):
    '''
    Given a subgraph G_i, finds the pendant set of 
    edges[p] intersect the pendant vertices of G^p_i.

    INPUT
    edges: The edges of the introduced node
    p: The index of the edge currently being added
    Gi: The subgraph G_i of G

    OUPUT
    pendant: A list of pendant vertices of G^p_i that are
    also in edges[p]
    '''
    Gpi = Gi.copy()

    #Remove the edges not in GPi
    for i in range(p, len(edges)):
        Gpi.remove_edge(*edges[i])

    pendant = set()

    #Go through all vertices and add pendant vertices
    vertices  = Gpi.nodes()

    for vertex in vertices:
        if nx.degree(Gpi, vertex) == 1:
            pendant.add(vertex)

    #Now find the ones also in edges[p]
    pendant = pendant.intersection(set(edges[p]))

    return pendant
    
def greatestInd(vertList, vertex):
    '''
    Given a list of sets of vertices, find the
    last index at which vertex appears.

    INPUT
    vertList: A list of sets of vertices
    vertex: The vertex that we are looking for

    OUTPUT
    last: The final index of vertex or inf if vertex is not
    in any sets of vertList
    '''
    last = float('inf')
    
    for i in range(len(vertList) - 1, -1, -1):
        
        #Find the last set with vertex
        if vertex in vertList[i]:
            last = i
            break

    return last

def leastInd(vertList, vertex):
    '''
    Given a list of sets of vertices, find the first
    index at which vertex appears.

    INPUT
    vertList: A list of sets of vertices
    vertex: A vertex to look for

    OUTPUT
    least: Either the least index at which vertex appears, or -1
    '''
    least = -1

    for i in range(0, len(vertList)):
        
        #Look for vertex in the set of vertList
        if vertex in vertList:
            least = i
            break

    return least

def sequenceUnion(seq1, seq2):
    '''
    Given two lists of sets of equal length returns
    the list that is the union of their sets (I_j U K_j).

    INPUT
    seq1: The first sequence of sets
    seq2: The second sequence of sets

    OUTPUT
    joined: A list of sets where each set is the union of the
    sets at the corresponding index in seq1 and seq2
    '''
    #If the sequences are of different length the algorithm went wrong
    if len(seq1) != len(seq2):
        raise Exception('Sequences are of different length!')
    
    joined = []

    for i in range(0, len(seq1)):
        joined.append(seq1[i].union(seq2[i]))

    return joined


def introduceEdge(char, j, m, edge, pendant):
    '''
    Computes a characteristic after a new edge is introduced in sequence
    j at position m.

    INPUT
    char: The characteristic to be modified
    j: The index of the sequence in which an edge is to be added
    m: The element at which the edge is to be added
    edge: The edge to be added
    pendant: The pendant edges in G^p_i

    OUTPUT
    charNew: The new characteristic after the edge is introduced
    '''
    #Copy the previous lists
    I = char[0]
    K = char[1]
    A = char[2]


    Inew = I[:]
    Knew = K[:]
    Anew = A[:]

    #Make a new position j in I
    Inew.insert(j + 1, Inew[j])

    #Subtract edge from the pendant set
    for i in range(0, len(Knew)):
        Knew[i] = Knew[i] - set(edge)

    #Add our new pendant set
    Knew.insert(j + 1, pendant)

    #Split the jth sequence of A at m
    firstSeq = Anew[j][:m + 1]
    secondSeq = Anew[j][m:]

    Anew[j] = firstSeq
    Anew.insert(j + 1, secondSeq)
    
    #Proceed to the insertion step
    for vertex in set(edge) - pendant:

        for h in range(min(greatestInd(sequenceUnion(Inew, Knew), vertex), j+1),
        max(j + 1, leastInd(sequenceUnion(Inew, Knew), vertex))):
            Inew[h] = Inew[h].union(vertex)
            
            #Increment all values in Anew at index h
            for i in range(0, Anew[h]):
                Anew[h][i] = Anew[h][i] + 1

    return [Inew, Knew, Anew]

def boundedBetween(start, end, sequence):
    '''
    Given a sequence of positive integers, checks
    whether the values between the start and end index 
    are bounded between those values.

    INPUT
    start: The index of the start value for the subsequence
    end: The index of the end value for the subsequence
    sequence: The sequence mentioned above

    OUTPUT
    bounded: True if bounded, False otherwise
    '''
    boundedBelow = True
    
    #Check if start val <= sequence[i] <= end val
    for i in range(start + 1, end):
        
        if not (sequence[start] <= sequence[i] 
        and sequence[i] <= sequence[end]):
            boundedBelow = False
            break

    if boundedBelow:
        bounded = True
        return bounded

    boundedAbove = True

    #Check if end  val <= sequence[i] <= start val
    for i in range(start + 1, end):
        
        if not (sequence[end] <= sequence[i] 
        and sequence[i] <= sequence[start]):
            boundedAbove = False
            break

    bounded = boundedAbove
    return bounded

def typicalSeq(sequence):
    '''
    Given a sequence of positive integers, outputs a
    typical sequence as described on pg. 5 of [1].

    INPUT
    sequence: A sequence of positive integers

    OUTPUT
    typical: The typical sequence corresponding to sequence
    '''
    typical = sequence[:]
    
    #Keep reducing until the sequence no longer changes
    changed = True

    while changed:
        changed = False
        
        #Check for repeats next to each other and remove
        for i in range(0, len(typical) - 1):
            
            if typical[i] == typical[i + 1]:
                changed = True
                typical = typical[:i + 1] + typical[i + 2:]
                break

        #Check for values bounded between two others
        for i in range(0, len(typical)):
            foundSub = False

            #Go through all the possible subsequences past i
            for j in range(i + 2, len(typical)):
                
                if boundedBetween(i, j, typical):
                    changed = True
                    foundSub = True
                    typical = typical[:i + 1] + typical[j:]
                    break
            
            #Stop iterating if we have modified typical
            if foundSub:
                break

    return typical

def compress(triple):
    '''
    Given a typical triple, runs the compression algorithm
    on pg. 6 of [1] to remove repeated values.

    INPUT
    triple: A typical triple [I, K, A]

    OUTPUT
    compressed: triple compressed with no repeat values
    '''

    I = triple[0][:]
    K = triple[1][:]
    A = triple[2][:]

    duplicates = True

    #Keep checking until we have gone through all values without duplicates
    while duplicates:
        
        foundDuplicate = False

        for h in range(0, len(I) - 1):
            Icurrent = I[h]
            Inext = I[h + 1]
            Kcurrent = K[h]
            
            #Check if the sets are equal and K_i+1 is empty
            if (Icurrent.issubset(Inext) and Inext.issubset(Icurrent)
            and len(Kcurrent) == 0):
                foundDuplicate = True
                
                #Remove the duplicate entry in I and K
                I = I[:h + 1] + I[h + 2:]
                K = K[:h + 1] + K[h + 2:]

                #Concat A_h with A_{h+1} and get the typical sequence
                A[h] = A[h] + A[h + 1]
                A[h] = typicalSeq(A[h])

                #Remove A[h + 1]
                A = A[:h + 1] + A[h + 2:]

                break
        
        if not foundDuplicate:
            duplicates = False

    return [I, K, A]

def maxSeq(seq):
    '''
    Given a sequence of sequence, find the maximum value in any
    of the sequences.

    INPUT
    seq: A sequence of sequences of positive integers

    OUTPUT
    maxVal: The maximum value found
    '''
    maxVal = seq[0][0]

    for sequence in seq:
        for value in sequence:
            
            if value > maxVal:
                maxVal = value

    return maxVal

def isEdge(graph):
    '''
    Given a graph, checks whether the graph is isomorphic to
    an edge. That is, it has two nodes that are joined by one
    edge.

    INPUT
    graph: A networkx graph

    OUTPUT
    edge: True if graph is an edge, False otherwise
    '''
    edge = False

    if (len(graph.nodes()) == 2 and 
    len(graph.edges()) == 1 and
    graph.has_edge(graph.nodes()[0], graph.nodes()[1])):
        edge = True

    return edge

firstEdge = True

def introduceNode(lastF, introduced, i, nicePath, G, k):
    '''
    Introduces a new node and calculates characteristics for it.
    See page 8 of [1]. Introduce-edge is implemented seperately
    for convenience.

    INPUT
    lastF: The full set of characteristics for G_{i-1}
    introduced: A set of length 1 containing the new vertex
    i: The current bag in the path
    nicePath: The nice path decomposition of G
    G: The networkx graph in question
    k: The k for which we are checking linear width

    OUTPUT
    FSi: The full set (in list form) of characteristics for G_i
    '''
    global firstEdge

    #Make subgraph G_i and find the edge set of our new node
    Gi = isubgraph(i, nicePath, G)

    newVertex = list(introduced)[0]
    edges = nx.edges(Gi, newVertex)

    #If there are no new edges, our FS has not changed
    if len(edges) == 0:
        FSi = lastF
        return FSi

    #Now we go edge by edge creating characteristics
    FSi = lastF

    for p in range(0, len(edges)):

        #Check if we are adding the first edge
        if firstEdge and p == 0:
            firstEdge = False
            FSi = [ [ [set(edges[0])], [set(edges[0])] , [[0]] ] ]
            continue

        newFS = []

        #Find the pendant vertices in G^p_i
        pendant = findPendant(edges, p, Gi)
        
        #Go through each characteristic and make new ones
        for char in FSi:
            
            I = char[0]
            A = char[2]

            for j in range(0, len(I)):
                for m in range(0, len(A[j])):
                    newChar = compress(introduceEdge(char, j, m, edges[p], pendant))
                    Anew = newChar[2]

                    #Check if the new characteristic has lin width <= k
                    if maxSeq(Anew) <= k:
                        newFS.append(newChar)

        FSi = newFS

    return FSi
            
def forgetNode(lastF, forgotten, i, nicePath, G):
    '''
    Implements Forget-Vertex from pg. 9 of [1]. Creates
    a new full set without the forgotten vertex.

    INPUT
    lastF: The full set of characteristics for G_{i-1}
    introduced: A set of length 1 containing the new vertex
    i: The current bag in the path
    nicePath: The nice path decomposition of G
    G: The networkx graph in question
    k: The k for which we are checking linear width

    OUTPUT
    FSi: The full set (in list form) of characteristics for G_i
    '''

    FSi = []

    for triple in lastF:
        Inew = triple[0][:]
        Knew = triple[1][:]
        A = triple[2][:]

        #Remove the forgotten vertex from I and K
        for i in range(0, len(Inew)):
            Inew[i] = Inew[i].difference(forgotten)
            Knew[i] = Knew[i].difference(forgotten)

        FSi.append(compress([Inew, Knew, A]))

    return FSi

def checkLinearWidth(G, k):
    '''
    Given a graph, calculates if the linear width is at 
    most k.
    See page 9 of [1].

    INPUT
    G: A networkx graph
    k: A positive integer against which linear width is checked

    OUTPUT
    atMost: Whether the linear width of G is at most k
    '''
    
    #Make a nice path decomposition of 
    nicePath = makeNicePathDecomp(G)
    print nicePath

    #Convert the bags of nicePath to sets
    for i in range(0, len(nicePath)):
        nicePath[i] = set(nicePath[i])
    
    #Initialize our full set of characteristics
    F = [ [[[],[],[]]] ]

    #Compute a new full set of characteristics for each node in nicePath
    for i in range(1, len(nicePath)):
        
        #Decide whether the node is introduce or forget
        introduced = nicePath[i] - nicePath[i - 1]
        forgotten = nicePath[i - 1] - nicePath[i]

        lastF = F[len(F) - 1]

        if len(introduced) == 1:
            
            #introduce a new vertex
            F.append(introduceNode(lastF, introduced, i, nicePath, G, k))
        elif len(forgotten) == 1:
            
            #forget a vertex
            F.append(forgetNode(lastF, forgotten, i, nicePath, G))

    #Check if the last characteristic is empty
    if len(F[len(F) - 1]) == 0:
        return False
    else:
        return True

G = nx.cycle_graph(4)
print checkLinearWidth(G, 1)
