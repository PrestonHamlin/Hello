#ifndef _BUBBLE_H
#define _BUBBLE_H

/* bubble.h - bubblesort sorting algorithm
This header and its associated .hpp file provide an implementation of the
  bubble-sort sorting algorithm.

*/

#include <vector>
#include <utility>    // swap, pair

// default of operator<
template<typename T>
std::pair<int, int> BubbleSort(std::vector<T> &array);

// custom comparator
template<typename T, typename compare>
std::pair<int, int> BubbleSort(std::vector<T> &array,
                               const compare& comp
                              );

// implementation
#include "bubble.hpp"


#endif
