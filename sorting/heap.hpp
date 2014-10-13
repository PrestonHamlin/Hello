using std::pair;
using std::vector;
using std::swap;


// helper function
template<typename T, typename compare>
pair<int, int> Heapify(vector<T> &array,
                       const compare& comp)
{

}


// default of operator<
template<typename T>
pair<int, int> HeapSort(vector<T> &array) {
  int accesses = 0;
  int comparisons = 0;
  typename vector<T>::iterator itr;

  for (itr = array.end(); itr != array.begin(); --itr) {
    make_heap(array.begin(), itr);  // rebuild heap, ignoring end elements
    pop_heap(array.begin(), itr);   // move head (max by default) to end
  }

  return pair<int, int> (comparisons, accesses);
}


// custom comparator
template<typename T, typename compare>
pair<int, int> HeapSort(vector<T> &array,
                        const compare& comp
                       )
{
  int accesses = 0;
  int comparisons = 0;

  return pair<int, int> (comparisons, accesses);
}
