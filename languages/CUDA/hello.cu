// Code by Preston Hamlin

// See header file for detailed comments.

#include "hello.h"




int main() {
  int cuda_status;
  int num_devs;
  //const int ARRAY_SIZE = 1<<20;   // one million elements
  
  // error code is cudaErrorNoDevice if no suitable devices exist
  if ((cuda_status = cudaGetDeviceCount(&num_devs)) != cudaSuccess) {
    FatalError("cudaGetDeviceCount", cuda_status);
  }
  else printf("\nDevices: %d\n", num_devs);

  int a[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  int* d_a  = 0;


  // attempt to allocate memory on device
  if ((cuda_status = cudaMalloc(&d_a, sizeof(int) * 16)) != cudaSuccess) {
    FatalError("cudaDeviceSynchronize", cuda_status);
  }
  cudaMemcpy(d_a, a, sizeof(int) * 16, cudaMemcpyHostToDevice);




  printf("\n=== Before ===\n");
  for (int i=0; i<16; ++i) printf("%d\n", a[i]);

  printf("\n=== After ===\n");
  ScanArrayBlelloch(d_a, 16, 0);
  cudaMemcpy(a, d_a, sizeof(int) * 16, cudaMemcpyDeviceToHost);
  for (int i=0; i<16; ++i) printf("%d\n", a[i]);






  // sync any remaining buffers (stdout for example)
  if ((cuda_status = cudaDeviceSynchronize()) != cudaSuccess) {
    FatalError("cudaDeviceSynchronize", cuda_status);
  }


  cudaFree(d_a);
  printf("\nHave a nice day.\n");
  return 0;
}






//============================================================================
//                      UserSpace Functions & Wrappers
//============================================================================

void FatalError(const char* msg, int e) {
  printf("\nERROR: %s\n", msg);
  if (e) printf("\tcode: %d\n", e);
  exit(1);
}
__host__ __device__ void NormalError(const char* msg, int e) {
  printf("\nERROR: %s\n", msg);
  if (e) printf("\tcode: %d\n", e);
}





//============================================================================
//                              CUDA Kernels
//============================================================================

// Simply prints the unique ID of each thread.
//  Used for checking the ordering of thread launch ranges
__global__ void CPrintID() {
  int blocknum, threadnum;
 
  blocknum =  blockIdx.x + 
              blockIdx.y*gridDim.x + 
              blockIdx.z*gridDim.x*gridDim.y;

  threadnum = threadIdx.x +
              threadIdx.y*blockDim.x + 
              threadIdx.z*blockDim.x*blockDim.y;

  printf("B%d/%d [%d/%d/%d] \t- T%d/%d [%d/%d/%d] \t- %d\n",
          blocknum,
          gridDim.x * gridDim.y * gridDim.z,
          blockIdx.x, blockIdx.y, blockIdx.z,

          threadnum,
          blockDim.x * blockDim.y * blockDim.z,
          threadIdx.x, threadIdx.y, threadIdx.z,

          blocknum*(blockDim.x * blockDim.y * blockDim.z) + threadnum
          );
}




