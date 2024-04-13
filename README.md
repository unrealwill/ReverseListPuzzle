# ReverseListPuzzle
Exploring the Reverse List Puzzle to brush-up graph-algorithms skills

The problem definition : https://mathstodon.xyz/@two_star/112242224494626411

"What does the sequence of maximal required moves look like as a function of n"

My code will allow you to compute 
The sequence up to [(5, 0), (6, 14), (7, 26), (8, 74), (9,86), (10,126)]
(If I didn't make mistakes :) )

For n=8 the hardest sequence is (1, 3, 5, 8, 7, 4)

It takes a few hours a few GB of memory for the last levels.

It uses brute-force, by computing the shortest path on the stategraph.

It's not optimized, i.e. single threaded, it's python with some compiled C routine.

It materializes a graph with nodes and edges using permutations and arrangement.
Then we can find the disjoint components, with union find.

We can then do some kruskal to have a spanning tree, from which we compute the diameter which is an upperBound of the searched quantity.

We can then compute the allpair distance for each of the disjoint components.

See the RevListPuzzle.py for more explanation

A few observations : 
- The graph is quite sparse, so message passing algorithms work well.
- Some invariant like the sum of all number is conserved by the rule, therefore there are various disjoint subgraphs.
- There is left-right symmetry.
- The Even and Odd sequence length have different behavior.
- The function is growing with increasing n
- The moves are reversible so the edges of the graph are undirected. 
