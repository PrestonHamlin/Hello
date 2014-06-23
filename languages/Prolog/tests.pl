% This is a sample directed graph for testing path finding.

% (1)-----(2)       Note that this is acually a directed graph.
%   \     /           1->2, 2->3, 3->1 - 123 is a 3-cycle
%    \   /            3->4, 3->5
%     (3)             4->6
%    /   \            7->5
%  (4)   (5)
%   |     |
%  (6)   (7)


%:- ['graph.pl'].


% n(1, [2]).
% n(2, [3]).
% n(3, [1, 4, 5]).
% n(4, [6]).
% n(5, []).
% n(6, []).
% n(7, [5]).

digraph([1,2,3,4,5,6,7],
        [e(1,2), e(2,3), e(3,1), e(3,4), e(3,5), e(4,6), e(7,5)]
       ).



run_tests :-
    X1 = [1,2,3,4,5,6,7],
    Y1 = [e(1,2), e(2,3), e(3,1), e(3,4), e(3,5), e(4,6), e(7,5)],

    write('Vertices: [1,2,3,4,5,6,7]'), nl,
    write('Edges:    [e(1,2), e(2,3), e(3,1), e(3,4), e(3,5), e(4,6), e(7,5)]'),
    nl, nl,

%    path(1, 1, 4, digraph(X1, Y1), P1),
%    write('1 to 1 in 4: '), write(P1), nl,

%    path(1, 4, 6, digraph(X1, Y1), P2),
%    write('1 to 6 in 4: '), write(P2), nl,

%    path(5, 4, 3, digraph(X1, Y1), P3),
%    write('5 to 4 in 3: '), write(P3), nl,

    path(3, 6, 10, digraph(X1, Y1), P4),
    write('3 to 6 in 10: '), write(P4), nl,

    write('\nHave a nice day.').


