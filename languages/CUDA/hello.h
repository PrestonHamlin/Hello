/*  Code by Preston Hamlin
This file provides some sample CUDA code. CUDA is used for GPGPU programming.
  The code is meant to be a "hello world" of sorts for CUDA, since the scan
  algorithm is one of the first one learns when exploring GPGPU.

The program contained herein provides for and demonstrates a scan algorithm.
  The scan (or prefix-sum) algorithm implemened here is the Blelloch variant.
  A scan algorithm takes an array of values (typically integers) and performs
  sucessive operations such that for each element a[i] in the array, a[i] is set
  to a[i-1] OP a[i]. This has the effect of a "running total."

  An example scan addition:
  before: [0, 1, 2, 3, 4,  5,  6,  7,  8,  9]
  after:  [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]

Here, a[0] is the start term, so it is OPed with the identity value, which is 0
  for addition. Meaning, a[0] = a[0] + 0, or a[0] = 0 + 0 = 0.
  The following are the remaining steps:
  a[1] = a[0] + a[1]    = 0 + 1   = 1
  a[2] = a[1] + a[2]    = 1 + 2   = 3
  a[3] = a[2] + a[3]    = 3 + 3   = 6
  a[4] = a[3] + a[4]    = 6 + 4   = 10
  a[5] = 10 + 5  = 15
  a[6] = 15 + 6  = 21
  a[7] = 21 + 7  = 28
  a[8] = 28 + 8  = 36
  a[9] = 36 + 45 = 45

  A max scan would maintain the maximum value encountered:
  before: [5, 12, 47, 16, 3,  48, 12, 55, 67, 3]
  after:  [5, 12, 47, 47, 47, 48, 48, 55, 67, 67]

One benefit of the concept is that algorithms exist for parallel computation.
  For any binary associative operator (addition, multiplication, union, etc...)
  the computations can be reordered and still yield the same result.
  1 + 2 + 3 + 4 would often be calculated by a human as
  ((1 + 2) + 3) + 4

  If the addition were to be right-associaive, however, it would look like
  1 + (2 + (3 + 4))

  Being of interest for parallel computation is
  (1 + 2) + (3 + 4)

  As is plainly visible, there are two components which can be computed 
  independently of one another. This independence resulting from the grouping
  of terms around an associative operator is what is utilized in the first half
  of the Blelloch algorithm. If only the end result is desired, then the first
  half will suffice, as the last element of the array will contain the result.
  For the cases where the intermediate results (the "running total") is also
  desired, the second step of the algorithm should also be utilized.

  Blelloch's algorithm used a divide-and-conquer approach to divide the array
  into a tree and then collect the value of lesser brances at intervals.

  Example run of Blelloch's scan algorithm with addition operator:
  0 1   1 0   1 1   1 0   0 1   0 1   1 1   0 1 
   \|    \|    \|    \|    \|    \|    \|    \|  
    1     1     2     1     1     1     2     1
      \   |       \   |       \   |       \   |
        \ |         \ |         \ |         \ |
          2           3           2           3
              \       |               \       |
                  \   |                   \   |
                      5                       5
                            \                 |
                                  \           |
                                        \     |
                                              10

  The elements in the array do indeed add up to 10. The elements in this array
  also happen to be the beginning of the Champernowne constant in binary, with
  a few bits cut off to have the array size be a power of 2.

  After the first step, the array looks like
  0 1   1 2   1 2   1 5   0 1   0 2   1 2   0 10

  This would be problematic for when the running total is needed. The second
  step of the algorithm takes the existing intermediate results and populates
  the remainder of the array. To facilitate the algorithm, the last term is set
  to 0 which is the identity value for addition.

  For each iteration, the right entry is moved to the left, and the left entry
  is OPed with the right entry and moved to the right.
  a[i] = right            a[7]  = 0
  a[j] = left + right     a[15] = 5 + 0
  
  0 1   1 2   1 2   1 5   0 1   0 2   1 2   0 0
                       \                     /
                            \           /
                                  X
                            /           \
                       /                     \
  0 1   1 2   1 2   1 0   0 1   0 2   1 2   0 5
           \         /             \         /
                X                       X
           /         \             /         \
  0 1   1 0   1 2   1 2   0 1   0 5   1 2   0 7
     \   /       \   /       \   /       \   /
       X           X           X           X
     /   \       /   \       /   \       /   \
  0 0   1 1   1 2   1 4   0 5   0 6   1 7   0 9
   X     X     X     X     X     X     X     X
  0 0   1 2   2 3   4 5   5 5   6 6   7 8   9 9

  Note that the resulting array seems a bit off. This is because Blelloch's
  algotithm is an exclusive scan, meaning that the value at each position in
  the result array reflects the running total up to that point not including
  that spot in the original data array. At index 2 there has been only one bit
  (remember that this is Champernowne's constant in binary) seen at index 1 so
  the exclusive scan has a value of 1 at that index. Similiarly at index 11 the
  sum of all the prior values is 6.

  To convert an exclusive scan to an inclusive one (that is, one representing
  the running total up to and including the current index) one can simply shift
  all values in the array to the left by one space and replace the last element
  in the array with the final total. One can either parse out the total before
  starting the second half of the algorithm, or hold on to the last element in
  the original data array and sum it with the second-to-last element of the
  result array.

  TODO: second half of Blelloch algorithm

  TODO: wrapper functions
    TODO: optimize thread usage
    TODO: split oddly sized arrays into power-of-two chunks

  TODO: version of scan optimized for latency/occupancy/perormance

  TODO: exceptions

  TODO: better options for min/max literals?
    TODO: try using d_ScanOpReduce with opposite operation to get identity val

*/


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>

#include <stdio.h>
#include <stdlib.h>


#define SCAN_OP_ADD 0
#define SCAN_OP_SUB 1
#define SCAN_OP_MUL 2
#define SCAN_OP_MIN 3
#define SCAN_OP_MAX 4
#define SCAN_OP_XOR 5


// declarations
void FatalError(const char* msg, int e=0);
__host__ __device__ void NormalError(const char* msg, int e=0);
__global__ void CPrintID();


// templated declarations
template <class T>
__device__ void d_ScanApplyOp(T* x, T* y, int mode);
template <class T>
__global__ void d_ScanOpReduce(T* d_a, int iters, int mode);
template <class T>
__global__ void d_ScanPrepArray(T* d_a, int iters, int mode);





//============================================================================
//                          Templated Functions
//============================================================================

/* d_ScanOpReduce - Reduce half of a scan operation
This is the first half of a scan operation. If only the resulting total or
  final value is desired, then one can use this without the second half. Else
  the other half should be used to populate the remainder of the array with
  the intermediate values.

  T* d_a    - some numeric type data array
  int iters - number of iterations to perform op, equal to lg of the array size
  int mode  - integer value describing operation (add, max, XOR, etc...) to use

  TODO: multi-block synchonization?
*/
template <class T>
__global__ void d_ScanOpReduce(T* d_a, int iters, int mode) {
  const int size = (1 << iters);
  int loc = 0, loc1 = 0, loc2 = 0;

  if (!d_a) {
    NormalError("NULL array passed to d_ScanOpReduce");
    return;
  }


  loc = threadIdx.x +
        threadIdx.y*blockDim.x + 
        threadIdx.z*blockDim.x*blockDim.y;

  for (int i=0; i<iters; ++i) {
    loc1 = (((2*loc)+1) * (1<<i)) - 1;
    loc2 = ((2*(loc+1)) * (1<<i)) - 1;

    if (loc2 < size) {  // if valid desination, perform op
      printf("%d - %d -> %d\n", i, loc1, loc2);
      
      switch (mode) {
        case SCAN_OP_ADD:
          d_a[loc2] = (d_a[loc1] + d_a[loc2]);
          break;
        // subtraction is simply the first term minus all others 
        //  negate d_a[0], scan add, negate all elements in array
        case SCAN_OP_SUB: 
          d_a[loc2] = (d_a[loc1] + d_a[loc2]);
          break;
        case SCAN_OP_MUL:
          d_a[loc2] = (d_a[loc1] * d_a[loc2]);
          break;
        case SCAN_OP_MIN:
          d_a[loc2] = (d_a[loc1] > d_a[loc2]) ? d_a[loc2] : d_a[loc1];
          break;
        case SCAN_OP_MAX:
          d_a[loc2] = (d_a[loc1] < d_a[loc2]) ? d_a[loc2] : d_a[loc1];
          break;
        case SCAN_OP_XOR:
//          printf("%08X ^ %08X = %08X\n", d_a[loc1], d_a[loc2],
//                 d_a[loc1] ^ d_a[loc2]);
          d_a[loc2] = (d_a[loc1] ^ d_a[loc2]);
          break;
        default:
          NormalError("unknown scan op");
          break;
      }
    }
    __syncthreads();  // synchronizes threads within block
  }

}


/* d_ScanOpDownSweep - Down-sweep half of a scan operation
This is the second half of a scan operation. This function is airly useless
  without first using the d_ScanOpReduce function. 

  T* d_a    - some numeric type data array
  int iters - number of iterations to perform op, equal to lg of the array size
  int mode  - integer value describing operation (add, max, XOR, etc...) to use

  TODO: multi-block synchonization?
*/
template <class T>
__global__ void d_ScanOpDownSweep(T* d_a, int iters, int mode) {
  const int size = (1 << iters);
  int loc = 0, loc1 = 0, loc2 = 0;
  T tmp = 0;

  if (!d_a) {
    NormalError("NULL array passed to d_ScanOpDownSweep");
    return;
  }


  loc = threadIdx.x +
        threadIdx.y*blockDim.x + 
        threadIdx.z*blockDim.x*blockDim.y;

  for (int i=(iters-1); i>=0; --i) {
    loc1 = (((2*loc)+1) * (1<<i)) - 1;
    loc2 = ((2*(loc+1)) * (1<<i)) - 1;

    if (loc2 < size) {  // if valid desination, perform op
      printf("%d - %d -> %d\n", i, loc1, loc2);
      
      // save right value
      tmp = d_a[loc2];

      switch (mode) {
        case SCAN_OP_ADD:
          d_a[loc2] += d_a[loc1];
          break;
        case SCAN_OP_SUB: 
          d_a[loc2] += d_a[loc1];
        case SCAN_OP_MUL:
          d_a[loc2] *= d_a[loc1];
          break;
        case SCAN_OP_MIN:
 //         printf("min(%d, %d) - %d\n", tmp, d_a[loc1], 
 //                ((tmp > d_a[loc1]) ? d_a[loc1] : tmp));
          d_a[loc2] = (tmp > d_a[loc1]) ? d_a[loc1] : tmp;
          break;
        case SCAN_OP_MAX:
 //         printf("max(%d, %d) - %d\n", tmp, d_a[loc1], 
 //                ((tmp < d_a[loc1]) ? d_a[loc1] : tmp));
          d_a[loc2] = (tmp < d_a[loc1]) ? d_a[loc1] : tmp;
          break;
        case SCAN_OP_XOR:
          d_a[loc2] = (tmp ^ d_a[loc1]);
          break;
        default:
          NormalError("unknown scan op");
          break;
      }

      // move right value to the left
      d_a[loc1] = tmp;
    }
    __syncthreads();  // synchronizes threads within block
  }
}


/* d_ScanPrepArray - Helper function called before d_ScanOpDownSweep
This function simply preps an array for the second part of the Blelloch scan
  algorithm. It is to be called before d_ScanOpDownSweep.

  T* d_a    - some numeric type data array
  int iters - number of iterations to perform op, equal to lg of the array size
  int mode  - integer value describing operation (add, max, XOR, etc...) to use

  TODO: multi-block synchonization?
*/
template <class T>
__global__ void d_ScanPrepArray(T* d_a, int iters, int mode) {
  int size = 1<<iters;
  
  // prep data array - set last element to identity value
  switch (mode) {
    case SCAN_OP_ADD:
      d_a[size-1] = 0;
      break;
    case SCAN_OP_SUB: 
      d_a[size-1] = 0;
      break;
    case SCAN_OP_MUL:
      d_a[size-1] = 1;
      break;
    case SCAN_OP_MIN:     // best identity value would be +infinity
      d_a[size-1] = 10000;
      break;
    case SCAN_OP_MAX:     // best identity value would be -infinity
      d_a[size-1] = 0;
      break;
    case SCAN_OP_XOR:
      d_a[size-1] = 0;
      break;
    default:
      NormalError("unknown scan op");
      break;
  }
}