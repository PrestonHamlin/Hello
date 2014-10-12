#include <vector>
#include <utility>  // swap, pair

using std::pair;
using std::vector;
using std::swap;

template<typename T>
std::pair<int, int> BubbleSort(std::vector<T> &array) {
  bool sorted = false;
  int accesses = 0;
  int comparisons = 0;
  typename vector<T>::iterator itr;

  while (!sorted) {
    sorted = true;

    for (itr = array.begin() + 1; itr != array.end(); ++itr) {
      if (*itr < *(itr-1)) {
        swap(*itr, *(itr-1));
        sorted = false;
        accesses += 2;  // one access per element per swap
      }
    comparisons++;
    accesses += 2;  // one access per element per comparison
    }
  }

  return pair<int, int> (comparisons, accesses);
}
