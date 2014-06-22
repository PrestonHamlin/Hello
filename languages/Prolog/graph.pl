% This file provides graph processing utilities based on various models for
%   describing graphs. 

% Code by Preston Hamlin

%                           === Graph Modeling ===
% Adjacency List
% In this method of modeling, the graph is described by a list of 2-tuples
%   where the first element of each tuple is a unique vertex and the second
%   element is a list of all vertices the paired vertex is connected to. With
%   this method, by default all graphs are directed and undirected graphs are
%   simply a special case in which for all pairs of vertices u and v, if v is 
%   in the adjacency list of u then u is in the adjacency list of v.
%
%
% Vertex & Edge Enumeration
% In this model, a list of vertices is declared and another list for edges
%   follows. This is a simplistic model and is often considered an uncondensed
%   version of the adjacency list. Some data storage formats omit the vertex
%   list, as the vertices are implied by the edge list. This may lead to losing
%   singleton vertices, but in CAD programs for instance they contribute almost
%   nothing when not associated with the mesh at large.
%
%   (1)------->(2)------->(3)
%               ^          |
%               |          |
%               |          v
%   (4)<------>(5)<-------(6)------->(7)
%   
% Adjacency List  
%   ( 1, [2]    ) 
%   ( 2, [3]    )
%   ( 3, [6]    )
%   ( 4, [5]    )
%   ( 5, [2, 4] )
%   ( 6, [5, 7] )
%   ( 7, []     )
%
% Vertex & Edge Enumeration
%   Vertices    1 2 3 4 5 6 7
%   Edges       (1, 2) (2, 3) (3, 6) (4, 5) (5, 2) (5, 4) (6, 5) (6, 7) 
%       
%
%
%
%                       === Implementing in Prolog ===
% There exist several notations usable in Prolog for describing a graph which
%   draw from the above modeling schemes.
%
% Adjacency-List Form
% The graph is represented by an adjacency list.
%
% Edge-Clause Form
% Each edge is stated as a single clause (fact). This is very much the abridged
%   V&EE format, although one can enumerate the vertices in clauses as well so
%   as to capture singletons (Vertex-, Edge-Clause Form).
%
% Graph-Term Form
% The graph is encapsulated in one large object by implementing its definition
%   as a pair of two sets, vertices and edges.
%
%
%
% Adjacency-List Form
%   [ n(1,[2]), n(2,[3]), n(3,[6]), n(4,[5]), n(5,[2,4]), n(6,[5,7]), n(7,[]) ]
%
% Edge-Clause Form
%   edge(1, 2).
%   edge(2, 3).
%   edge(3, 6).
%   edge(4, 5).
%   edge(5, 2).
%   edge(5, 4).
%   edge(6, 5).
%   edge(6, 7).
%
% Graph-Term Form
%   digraph( [1, 2, 3, 4, 5, 6, 7],
%            [e(1,2), e(2,3), e(3,6), e(4,5), e(5,2), e(5,4), e(6,5), e(6,7)]
%          ).
%
%
%
%                               === Decisions ===
% Since there are several ways to construct a graph with additional information
%   such as edge direction and weight, several versions of processing functions
%   will be provided with appropriate suffixes:
%       path          - finds a finite path connecting two nodes
%       path_shortest - finds the shortest path
%       path_cheapest - finds least expensive path
%       path_optimal  - finds optimal path (hops have weight)
%   
% Some specialized functions will also be implemented. For example, the optimal
%   path from one node to another which must pass though a third node. Or a 
%   path which must bypass a node.
%
%
% 
% TODO: Helper functions
% 
% TODO: Pathing
%   TODO: Simple path
%   TODO: Shortest path     implement BFS
%   TODO: Cheapest path     unweighted graph has edge weights of 1
%   TODO: Optimal path      path length added to edge costs
%
%   TODO: Sidetracked       additional manditory nodes
%   TODO: Deadlines         Sidetracked, with deadlines on each node
%   TODO: Frugal Tourist    Sidetracked, with cheapest path
%   TODO: Tour Group        Combination of Deadlines and Frugal Tourist
%
% TODO: Find cycles
% TODO: Detect negative cycles
%   TODO: Stop and say -infinity
%





% The way Prolog attempts to solve queries is essentially a backtracking search
%   which will try alternate solutions to rules and alternate definitions of
%   the same rules. In rules with no terminal case or where there are no bounds
%   to growth, it behaves much like a depth-first-search on an infinite tree.
%
% This is problematic, since graphs usually have a multitude of ways to
%   traverse from one node to another and looping until the stack limit is hit
%   is hardly a solution. To remedy this, an upper bound is used on the number
%   of node transitions. It is easily provable that if two nodes are in the
%   same graph of size n, and at least one path exists between these nodes then
%   there exists a path connecting the nodes which is of length n-1 or less.
%
% As such, I am using wrapper functions which do not take a limit on path
%   length. These wrappers use the number of nodes in the graph as a default
%   and call the actual path-finding functions using this value as the upper
%   limit on path length. A more optimal approach would be to attempt some 
%   pre-processing to determine a better lower bound. However, like many 
%   heuristics, as the estimate gets better the heuristic solves more and more
%   of the problem itself. The best heuristic for finding an optimal weighted
%   path is comparing against the optimal path itself.



%==============================================================================
%                            Convertion Functions 
%==============================================================================



%==============================================================================
%                              Helper Functions  
%==============================================================================



%==============================================================================
%                             Pathing Functions 
%==============================================================================

