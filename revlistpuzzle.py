import itertools
import numpy as np
from cffi import FFI


ffi = FFI()
ffi.set_source("_test", """               
int graphDistTo( int i0, int* dj,int lendj, int* edges, int nedges)
{
    for(int i =0 ;i < lendj ; i++)
    {
        dj[i] = 1000000;//Positive infinity
    }
    
    dj[i0] = 0;
    
    int hasChanged = 1;
    int niter = 0;
    while( hasChanged == 1)
    {
     niter = niter+1;
     hasChanged = 0;
    //edges already contain reverse edges
    for( int i = 0 ; i < nedges ; i++)
    {
       int e0 = edges[2*i];
       int e1 = edges[2*i+1];
       int min = dj[e0];
       if( dj[e1] + 1 < min)
       {
         min = dj[e1]+1;
         hasChanged = 1;
       }
       dj[e0] = min;
       
                       
    }
    }
    return niter;
}

int batchGraphDistTo( int* ind,int* revind, int nbind, int* dj, int lendj, int* edges, int nedges )              
{
    int max = -1;
    for( int k = 0 ; k < nbind ; k++ )
    {
      int i0 = ind[k];
      int j0 = revind[k];
      if (i0 == j0) continue;
      graphDistTo(i0,dj,lendj,edges,nedges);
      if( dj[j0] > max)
      {
        max = dj[j0];
      }      
    }
    return max;
}

""")
ffi.cdef("""int graphDistTo(int, int*,int,int*,int);""")
ffi.cdef("""int batchGraphDistTo( int* ind,int* revind, int nbind, int* dj, int lendj, int* edges, int nedges );""")
ffi.compile()

from _test import lib     # import the compiled library



'''Copy pasted and adapted from https://www.geeksforgeeks.org/building-an-undirected-graph-and-finding-shortest-path-using-dictionaries-in-python/'''
# Code only use to check computation by brute-force
# Python implementation to find the 
# shortest path in the graph using 
# dictionaries 
 
# Function to find the shortest
# path between two nodes of a graph
def BFS_SP(graph, start, goal):
    explored = []
     
    # Queue for traversing the 
    # graph in the BFS
    queue = [[start]]
     
    # If the desired node is 
    # reached
    if start == goal:
        #print("Same Node")
        return list(start)
     
    # Loop to traverse the graph 
    # with the help of the queue
    while queue:
        path = queue.pop(0)
        node = path[-1]
         
        # Condition to check if the
        # current node is not visited
        if node not in explored:
            neighbours = graph[node]
             
            # Loop to iterate over the 
            # neighbours of the node
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                 
                # Condition to check if the 
                # neighbour node is the goal
                if neighbour == goal:
                    #print("Shortest path = ", *new_path)
                    return new_path
            explored.append(node)
 
    # Condition when the nodes 
    # are not connected
    #print("So sorry, but a connecting path doesn't exist :(")
    return list()

def shortest_path(graph, node1, node2):
    path_list = [[node1]]
    path_index = 0
    # To keep track of previously visited nodes
    previous_nodes = {node1}
    if node1 == node2:
        return path_list[0]
        
    while path_index < len(path_list):
        current_path = path_list[path_index]
        last_node = current_path[-1]
        next_nodes = graph[last_node]
        # Search goal node
        if node2 in next_nodes:
            current_path.append(node2)
            return current_path
        # Add new paths
        for next_node in next_nodes:
            if not next_node in previous_nodes:
                new_path = current_path[:]
                new_path.append(next_node)
                path_list.append(new_path)
                # To avoid backtracking
                previous_nodes.add(next_node)
        # Continue to next path in list
        path_index += 1
    # No path is found
    return []

"""End copy paste"""



class MySet:
    def __init__(self, v) :
        self.parent = None
        self.v = v
        self.rang = 0

    @staticmethod
    def find( x ):
        if x.parent == None:
            return x
        return MySet.find( x.parent )
    
    @staticmethod
    def union( x,y):
        xrac = MySet.find( x )
        yrac = MySet.find( y )
        if xrac != yrac :
            if xrac.rang < yrac.rang:
                xrac.parent = yrac
            else:
                yrac.parent = xrac
                if xrac.rang == yrac.rang:
                    xrac.rang = yrac.rang + 1 

    @staticmethod
    def areJoined( x,y):
        return MySet.find(x) == MySet.find(y)

class RevListPuzzle:
    def __init__(self,n) :
        digits = [ x+1 for x in range(n)]
        count = 0
        self.n = n
        self.nodes = []

        for k in range(n):
            ii = 0
            for p in itertools.combinations(digits,k):
                for p2 in itertools.permutations(p ):
                    self.nodes.append(p2)
                    ii = ii +1
                    count = count+1
                    if count % 1000000 == 0:
                        print("k : " + str(k) + " count : " + str(count))

        print("Number of nodes : " + str(count))

        self.nodedict = {}

        for node in self.nodes:
            self.nodedict[ node ] = node

        #We can materizalize the edges
        self.computeLabelUnionFind()

        #self.computeLabels()

    def contractingEdges( self, node, n ):
        tuprep = node
        out =[]
        for p in range(len(tuprep)-1):
            sum = tuprep[p]+tuprep[p+1]
            if( sum <= n ):
                tup = tuprep[0:p] + (tuprep[p]+tuprep[p+1],) + tuprep[p+2:]
                #if( tup in nodedict):
                #    out.append( nodedict[tup]  )
                if( tup in self.nodedict):#To remove distinct value we check if it is in the dictionary
                    out.append(tup)
        return out

    def dfs( self, graph, node,visited, visit, depth):
        visit(node,depth)
        visited[node] = True
        for nod in graph[node]:
            if( nod not in visited):
                visit(nod,depth)
                visited[nod] = True
                self.dfs( graph, nod, visited, visit, depth+1 )


    def height( self, tree, root, visited):
        visited[root] = True
        curmax = 0

        for node in tree[root]:
            if node not in visited:
                heightchild = self.height(tree,node,visited)
                if heightchild > curmax:
                    curmax = heightchild
        return 1 + curmax
            

    def diameter( self, tree , root, visited):
        visited[root]=True

        visitheight = {}
        visitheight[root]=True
        heightsOfChildren = [0,0]
        for node in tree[root]:
            if node not in visited:
                heightsOfChildren.append( self.height( tree, node,visitheight ) )
        
        heightsOfChildren.sort(reverse=True)

        diametersOfChildren = [0]
        for node in tree[root]:
            if node not in visited:
                diametersOfChildren.append( self.diameter(tree,node,visited))

        return max( heightsOfChildren[0]+heightsOfChildren[1], max(diametersOfChildren))

    def computeDiameterTree( self, tree):
        root = next(iter(tree))
        visited = {}
        diam = self.diameter(tree,root,visited)
        return diam


    def kruskal( self , graph):
        spanningTree = {}
        kruskalset = {}
        for node in graph:
            kruskalset[ node ] = MySet( None )

        for node in graph:
            for neighbor in self.contractingEdges(node,self.n):
                u = kruskalset[node]
                v = kruskalset[neighbor]
                if( MySet.find(u) != MySet.find(v) ):
                    if node in spanningTree :
                        spanningTree[node].append(neighbor)
                    else:
                        spanningTree[node]= [neighbor]
                    if neighbor in spanningTree :
                        spanningTree[neighbor].append(node)
                    else:
                        spanningTree[neighbor ] = [node]

                    MySet.union(u,v)
        return spanningTree


    # #https://www.osti.gov/servlets/purl/1474328
    #"Computing Exact Vertex Eccentricity on Massive-Scale Distributed Graphs"

    def computeLabelUnionFind( self ):
        self.mylabels = {}
        for i in range(len(self.nodes)):
            self.mylabels[ self.nodes[i] ] = MySet( i )
        for node in self.nodes:
            for neighbor in self.contractingEdges(node,self.n):
                MySet.union( self.mylabels[node],self.mylabels[neighbor])
        self.labels = {}
        for i in range(len(self.nodes)):
            self.labels[ self.nodes[i] ] = MySet.find( self.mylabels[self.nodes[i]] ).v
        
        self.ccbylabel = {}
        for i in range(len(self.nodes)):
            lab = MySet.find( self.mylabels[self.nodes[i]] ).v
            if lab not in self.ccbylabel:
                self.ccbylabel[lab] = [self.nodes[i]]
            else:
                self.ccbylabel[lab].append(self.nodes[i])

        #print("Labels : ")
        #print(self.labels)
        #print( "connected components")
        #print( self.ccbylabel )
        print( "len(self.ccbylabel )")
        print( len(self.ccbylabel) )

        bigSubgraphs = [ val for key,val in self.ccbylabel.items() if len(val) > 100]
        print( "number of subgraph of length > 100 : " + str(len(bigSubgraphs)))

    def computeMaxDiameterOfCCspanningTrees( self):
        diameters = []
        for lab,cc in self.ccbylabel.items():
            if len(cc) > 1:
                st = self.kruskal(cc)
                diameters.append(self.computeDiameterTree( st ) )
        print("diameters of spanning trees")
        print( diameters )

        diameters.append(0)
        return max(diameters)

    def buildEdges( self ):
        edges = [ (node, neighbor) for node in self.nodes for neighbor in self.contractingEdges(node,self.n) ]
        print("len(edges)")
        print( len(edges))
        return edges

    def buildFullGraph(self, edges):
        graph = {}

        for node in self.nodes:
            graph[ node ] = []

        for e in edges:
            graph[ e[0] ].append(e[1])
            graph[ e[1] ].append(e[0])
        return graph

    def buildUndirectedSubgraph(self, node ):
        graph = {}
        lab = self.labels[ node ]
        cc = self.ccbylabel[lab]

        for nod in cc:
            graph[nod] = []

        for nod in cc:
            edges = self.contractingEdges(nod,self.n)
            for ee in edges:
                graph[ nod ].append(ee)
                graph[ ee ].append(nod)
        return graph


    def pairwiseDistanceInCC( self, CC, graph ): 
        scc = len(CC)
        

        ccdict = {}

        for i in range(scc):
            ccdict[CC[i]]= i

        """
        #Initialisation for floyd warshall
        for inode in range(scc):
            node = CC[inode]
            for neighbor in graph[node]:
                inn = ccdict[neighbor]
                dij[inode, inn ] = 1
        """
        #Compute all pairwise distance for connected component with floyd warshall
        #for k in range(scc):
        #    dij = np.minimum(dij,dij[:,k:(k+1)]+dij[k:(k+1),:] )


        #Compute all pairwise distance by message passing to make use of the sparsity
        direvi = [-1] 
        '''
        #Alternatively we can bruteforce the shortest_path
        for i in range(scc):
            if( i % 1000) == 0:
                print(i)
            revtup = tuple(reversed(CC[i]))
            direvi.append(len(shortest_path(graph,CC[i],revtup))-1  )
        '''
        
        inedges = []
        for j in range(scc):
            for e in graph[CC[j]]:
                if( e in ccdict):
                    inedges.append((j,ccdict[e]))

        #print(inedjk.shape)
        dijarr = np.zeros((scc,),dtype=np.int32)
        dij_arr = ffi.cast('int*', dijarr.ctypes.data)
        edges = np.array(inedges,dtype=np.int32)
        edges_arr = ffi.cast('int*', edges.ctypes.data)

        bs = 1000
        indarr = np.zeros((bs,),dtype=np.int32)
        ind_arr = ffi.cast('int*', indarr.ctypes.data)
        revindarr = np.zeros((bs,),dtype=np.int32)
        revind_arr = ffi.cast('int*', revindarr.ctypes.data)

        if( scc > 1000):
            print("len(inedges)")
            print(len(inedges))

        for i in range(0,scc,bs):
            if( i >=1000 ):
                print( "pairwiseDistanceInCC row "+str(i)+ " / " +str(scc))
            nbind = min(bs, scc-i)
            for j in range(nbind):
                revtup = tuple(reversed(CC[i+j] ) )
                indarr[j] = i+j
                revindarr[j] = ccdict[revtup] if revtup in ccdict and self.n in CC[i+j] else i+j
            maxdirevi = lib.batchGraphDistTo(ind_arr,revind_arr,nbind,dij_arr,scc,edges_arr,edges.shape[0])
            direvi.append(maxdirevi)
            
        '''
        #Non batched version
        for i in range(scc):
            #We can skip the starting sequence that don't contain n as we don't care about the result
            if( self.n in CC[i]):
                niter = lib.graphDistTo( i, dij_arr,scc, edges_arr,edges.shape[0] ) 
            #Typically due to sparsity : niter ~30 << scc 

            revtup = tuple(reversed(CC[i] ))
            #We require that n is in the starting state
            if( revtup in ccdict and self.n in CC[i]):
                distirevi = dijarr[ccdict[revtup]]
                direvi.append( distirevi )
            else:
                direvi.append(-1)
            if (i+1) % 1000 ==0:
                print( "pairwiseDistanceInCC row "+str(i+1)+ " / " +str(scc))
        '''
        return direvi

    def computeDistItoRevi(self):
        distinctlabel = list( set( [val for _,val in self.labels.items() ]) )

        maxdirevibydistinctlabel = {}
        maxdirevibydistinctlabel[-1] = -1
        print("Grouping nodes by label")
        nodesByLabel = {}
        for node in self.nodes:
            lab = self.labels[node] 
            if lab in nodesByLabel:
                nodesByLabel[lab].append(node)
            else:
                nodesByLabel[lab] = [node]

        print("Grouping nodes by label done")

        CCSizeByLab = {}
        for lab,CC in nodesByLabel.items():
            CCSizeByLab[lab] = len(CC)

        sortedLabs = dict(sorted(CCSizeByLab.items(), key=lambda item: -item[1] ))

        #sortedByCCSizeNodesByLabel = dict(sorted(nodesByLabel.items(), key=lambda item: -len(item[1])))


        ilab = 0
        currentMaxdirevi = 0

        for lab,lenCC in sortedLabs.items():
            CC = nodesByLabel[lab]
            if( len(CC) <= currentMaxdirevi ):
                break

            if( len(CC) > 100):
                print("len(CC)")
                print(len(CC))
            if( ilab % 1000 == 0):
                print("ilab : "+ str(ilab) + " / " + str(len(nodesByLabel)))
            ilab = ilab+1
            #CCinterRevCC = [node for node in CC if labels[ tuple(reversed(node)) ]==lab ]
            #print("len(CCinterRevCC) : " + str(len(CCinterRevCC)) )
            subgraph = self.buildUndirectedSubgraph(CC[0])

            direvi = self.pairwiseDistanceInCC( CC ,subgraph)
            maxdirevi = max(direvi)
            if maxdirevi < 100000 and maxdirevi > currentMaxdirevi:
                currentMaxdirevi = maxdirevi

        print("length longest shortest path between seq and revseq")
        print( currentMaxdirevi )
        return currentMaxdirevi


    #Brute forcing to check solution

    #allrevdistancesAndPath =[ BFS_SP( graph, node, tuple( reversed(node) ) ) for node in nodes 
    #                                    if n in node and labels[node] == labels[tuple( reversed(node) )]  ]

    #allrevdistancesAndPath =[ shortest_path( graph, node, tuple( reversed(node) ) ) for node in nodes 
    #                                    if n in node and labels[node] == labels[tuple( reversed(node) )]  ]

    def computeAllDistanceToRev(self, fullgraph ):
        allrevdistancesAndPath = []

        co= 0
        for inode in range(len(self.nodes)):
            node = self.nodes[inode]
            co = co+1
            if( co%1000 == 0):
                print("checking shortest path node " + str(co) + " / " + str(len(self.nodes)))
            if self.n in node :
                spntorevn = shortest_path( fullgraph, node, tuple( reversed(node) ) )
                #if( len(spntorevn) > 0 ):
                #    print(spntorevn)
                allrevdistancesAndPath.append(spntorevn)


        #maxallrevdistance = max([d for d,path in allrevdistancesAndPath] )
        longestShortestPath = []
        maxd = -1
        for path in allrevdistancesAndPath:
            d = len(path) - 1
            if d > maxd:
                maxd = d
                longestShortestPath = path

        print("longest shortest path :")
        print(longestShortestPath)

        print("length of longest path : " + str(maxd))

        print("verification of length of shortest path ")
        start = longestShortestPath[0]
        end = longestShortestPath[-1]
        sp = shortest_path( fullgraph,start,end)
        print(len(sp)-1)
        return (len(sp)-1)


    #We allow to go with values up to n, not necessarily up to max(start)
    def getSolution( self,graph, seq ):
        if self.n not in seq:
            print( "Warning n not in seq")
        return shortest_path( graph, seq, tuple(reversed(seq)))


def demo1():
    print("demo1()")
    revlistpuzzle = RevListPuzzle( 8 )
    
    spanningTree =revlistpuzzle.kruskal( revlistpuzzle.buildUndirectedSubgraph( (3,5,7)) )
    print("spanningTree of connected component of (3,5,7) ")
    print(spanningTree)
    maxDiameterOfCCspanningTrees = revlistpuzzle.computeMaxDiameterOfCCspanningTrees()
    print("UpperBound maxDiameterOfCCspaningTrees : ")
    print(maxDiameterOfCCspanningTrees)

    revlistpuzzle.computeDistItoRevi()

    fullgraph = revlistpuzzle.buildFullGraph( revlistpuzzle.buildEdges() )
    revlistpuzzle.computeAllDistanceToRev(fullgraph)

    print( "revlistpuzzle.getSolution( (3,5,7) )" )
    print( revlistpuzzle.getSolution(fullgraph, (3,5,7) )) 

def demoPlot():
    print("demoPlot()")
    plotting = [ (i,RevListPuzzle(i).computeDistItoRevi()) for i in range(5,9,1)]
    print( plotting )

def demoUnionFind():
    print("demoUnionFind()")
    a = MySet( (3,4,5) )
    b = MySet( (4,3,5) )
    c = MySet( (4,3,6) )
    d = MySet( (4,3,6) )

    MySet.union(a,b)
    print(MySet.areJoined(a,d))
    MySet.union(c,d)
    MySet.union(c,b)
    print(MySet.areJoined(a,d))

    print(a.rang)
    print(b.rang)
    print(c.rang)
    print(d.rang)

demoUnionFind()
demo1()
demoPlot()