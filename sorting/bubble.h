#ifndef _BUBBLE_H
#define _BUBBLE_H

/* bubble.h - bubblesort sorting algorithm
This header and its associated .hpp file provide an implementation of the
  bubble-sort sorting algorithm.

TODO: template takes custom comparator
*/

#include <vector>
#include <utility>          // swap, pair


template<typename T>
std::pair<int, int> BubbleSort(std::vector<T> &array);


// implementation
#include "bubble.hpp"


#endif
