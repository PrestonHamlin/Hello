/* sorts - collection of sorting algorithms
This file contains the main routine which demonstrates various algorithms for
  the purposes of sorting a sequential array of comparable objects.

TODO: large array demo
TODO: other sorts
  TODO: heap
  TODO: insertion
  TODO: merge
  TODO: quick
  TODO: radix
  TODO: selection
*/

#include "sorts.h"

using std::cout;
using std::pair;
using std::vector;

int main() {
  vector<int> foo {83, 53, 28, 12, 74, 15, 72, 16, 62, 35};
  vector<int> foo_bubble(foo);

  pair<int, int> results_bubble;

  // lambda function to demo templated comparison function
  auto gt_lambda = [](int a, int b) {return a>b;};


  // perform bubble sort
  results_bubble = BubbleSort(foo_bubble);
  cout << "\n\n=== Bubble Sort ==="
       << "\n  comparisons:    " << results_bubble.first
       << "\n  array accesses: " << results_bubble.second
       << "\n";
  for (auto val : foo_bubble) cout << val << " ";


  cout << "\n\nHave a nice day\n";
  return 0;
}
