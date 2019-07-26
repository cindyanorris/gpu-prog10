#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "d_scan.h"
#include "config.h"  //defines THREADSPERBLOCK
#include "wrappers.h"


static __global__ void sumKernel(int *, int *, int);
static __global__ void sweepKernel(int *, int *) ;
static void cpuScan(int *, int);
static void exclusiveScan(int *, int, int);
__device__ void gpuPrintVec(const char * label, int * vector, int length);
void cpuPrintVec(const char * label, int * vector, int length);

/* d_scan
 * This function is a wrapper for the exclusive scan that is 
 * performed on the GPU. It uses cudaMalloc to create an input/output
 * array on the GPU and copies the CPU array to the GPU array. 
 * It initializes the timing functions and then calls the 
 * exclusiveScan function to do the scan.
 * You should not modify this function.
 *
 * @param - output contains a pointer to the array to hold
 *          the output of the scan when complete
 * @param - input contains a pointer to the input to the scan 
 * @param - length is the size of the output array
 * @param - numEles is the number of elements to partition
 *          to a thread to complete the final sum for the scan
 */
float d_scan(int * output, int * input, int length, int numEles)
{
    int * d_output;
    float cpuMsecTime = -1;
    cudaEvent_t start_cpu, stop_cpu;

    //THIS FUNCTION IS COMPLETE

    //To reduce the amount of time spent doing memory allocations,
    //create a single input/output array for GPU
    CHECK(cudaMalloc((void **)&d_output, sizeof(int) * length));
    CHECK(cudaMemcpy(d_output, input, length * sizeof(int), 
                     cudaMemcpyHostToDevice));

    //start the timing
    CHECK(cudaEventCreate(&start_cpu));
    CHECK(cudaEventCreate(&stop_cpu));
    CHECK(cudaEventRecord(start_cpu));

    //do the scan and wait for all threads to complete
    exclusiveScan(d_output, length, numEles);
    cudaThreadSynchronize();

    //stop the timing
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu));
    CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));

    //copy the output of the GPU to the CPU array
    cudaMemcpy(output, d_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    //release the space for the GPU array
    CHECK(cudaFree(d_output));

    return cpuMsecTime;
}

/*
 * exclusiveScan
 * Launches two kernels to performs the exclusive scan on the GPU
 *
 * @param - d_output array that contains the input
 *          to use for the exclusive scan and holds
 *          the output of the exclusive scan
 * @param - length of array; always a power of 2
 * @param - numEles is the number of elements to partition
 *          to a thread to complete the scan; must be
 *          a power of 2 and less than or equal to
 *          MAXELES
 *          
*/
void exclusiveScan(int * d_output, int length, int numEles)
{
   if (length <= THREADSPERBLOCK)
   {
      //If the length is less than or equal to the block size then
      //do the scan on the CPU. Do this by implementing
      //and calling the cpuScan function. 
      //The result of this scan needs to be stored in
      //d_output.  Note d_output is a pointer to data
      //in GPU memory.  To use the CPU function, you'll need to copy
      //that data into CPU memory, call the CPU function,
      //and then copy the result back into GPU memory.  You can
      //test this function without implementing any of the other
      //code by running the program on vectors who are not
      //greater than THREADSPERBLOCK in length.  For example,
      // ./scan -s 9

      /* put the code to do what was described above here */

      return;
   } else
   {
      //1) Launch the sweepKernel
      //THREADSPERBLOCK is defined; Use length and THREADSPERBLOCK
      //to define the grid.
      //The kernel needs to be passed an array S
      //that will hold the values to be added to complete
      //the partial scan in d_output. The size of S will be
      //dependent upon the size of the grid.
      //You'll need to define and create S. The kernel is
      //provided; you just need to launch it.

      /* missing code goes here */

      //2) Don't continue on until all of the threads terminate
      CHECK(cudaDeviceSynchronize());

      //3) Perform an exclusive scan on S (by calling exclusiveScan)
      //Isn't recursion great?
      //If S is 1 2 3 4 then exclusiveScan sets it to 0 1 3 6 

      /* missing code goes here */

      //4) Write and launch the sumKernel to add the elements of
      //S to elements of d_output. Use same grid dimensions that you defined
      //before, but set the number of threads in a block to
      //THREADSPERBLOCK/numEles.  This way each block i in this
      //kernel launch operates on the same elements as block i in
      //the sweepKernel kernel launch, but may use fewer threads
      //to do the work (depending upon the value of numEles).  

      /* missing code goes here */
   }
}


/*
 * sweepKernel
 * Performs an exclusive scan on the data on the d_output
 * array. In addition, one thread in each block will set an 
 * element in the sum array to the value that needs to be
 * added to the elements in the next section of d_output
 * to complete the scan.
 *
 * @param - d_output points to an array in the global memory
 *          that holds the input and will be modified to hold
 *          the output
 * @param - sum points to an array to hold the value to be 
 *          added to the section handled by blockIdx.x + 1
 *          in order to complete the scan
*/
__global__ void sweepKernel(int * d_output, int * sum)
{
   //THIS FUNCTION IS COMPLETE
   
   __syncthreads();
   int tid = threadIdx.x;
   int blkD = blockDim.x;
   int blkI = blockIdx.x;

   //d_input points to the section of the input to be
   //handled by this block
   int * d_input = d_output + blkI * blkD;
   __shared__ int shInput[THREADSPERBLOCK];

   //initialize the value in the sum array
   if (tid == (blkD >> 2) - 1)
   {
      sum[blkI] = d_input[blkD - 1];
   }

   //all threads participate in loading a
   //value into the shared memory
   shInput[tid] = d_input[tid];

   __syncthreads();
   int thid0Index = 0;
   int index;
   for (int i = 1; i < blkD; i<<=1)
   {
      thid0Index = thid0Index + i; 
      index = thid0Index + tid * 2 * i;
      if (index < blockDim.x) 
      {
         shInput[index] += shInput[index-i];
      }
      __syncthreads();
   }
  
   //set the last element in the section to 0 
   //before the next sweep
   if (tid == (blkD >> 2) - 1) shInput[blkD - 1] = 0;
   __syncthreads();  
   int i, j, topIndex, botIndex, tmp;
   for (j=1, i = blkD >> 1; i >= 1; i >>= 1, j <<= 1)
   {
      //first iteration thread 0 is active
      //second iteration threads 0, 1 are active
      //third iteration threads 0, 1, 2, 4
      if (tid < j)
      {
         topIndex = (tid + 1) * 2 * i - 1;
         botIndex = topIndex - i;
         tmp = shInput[botIndex];
         shInput[botIndex] = shInput[topIndex];
         shInput[topIndex] += tmp;
      }
      __syncthreads();
   }
   d_input[tid] = shInput[tid];
   //update sum using last element in the block
   if (tid == (blkD >> 2) - 1) sum[blkI] += shInput[blkD - 1];

   __syncthreads();
}

/*
 * sumKernel
 * Adds elements in sum to the elements in the d_output array.
 * The elements in the d_output array are sectioned into chunks
 * of size THREADSPERBLOCK.  sum[0] is added to the first chunk.
 * sum[1] added to the second block, etc.  The work is partitioned
 * among the threads in the block using cyclic partitioning.  Each thread
 * computes numElements results.
 * @param - sum points to an array of values to use to update
 *          d_output
 * @param - d_output points to the array of partially scanned
 *          values
*/
__global__ void sumKernel(int * sum, int * d_output, int numElements)
{
   //YOU NEED TO WRITE THIS
}

/* 
 * cpuScan
 * Performs an exclusive scan operation on the CPU.
 * @param - vector is a pointer to array of integers to
 *          scan
 * @param - length is the number of elements in an array
 * @modifies - vector array
*/
void cpuScan(int * vector, int length)
{
   //YOU NEED TO WRITE THIS
   //It isn't the same as the one in the book. That 
   //does an inclusive scan.

}

/* 
 * gpuPrintVec
 * Prints the contents a vector that is in the GPU memory, 10 elements
 * per line.  This can be used for debugging.
*/
__device__ void gpuPrintVec(const char * label, int * vector, int length)
{
    int i;
    printf("%s", label);
    for (i = 0; i < length; i++)
    {
        if ((i % 10) == 0) printf("\n%4d: ", i);
        printf("%3d ", vector[i]);
    }
    printf("\n");
}

/* 
 * cpuPrintVec
 * Prints the contents a vector that is in the CPU memory, 10 elements
 * per line.  This can be used for debugging.
*/
void cpuPrintVec(const char * label, int * vector, int length)
{
    int i;
    printf("%s", label);
    for (i = 0; i < length; i++)
    {
        if ((i % 10) == 0) printf("\n%4d: ", i);
        printf("%3d ", vector[i]);
    }
    printf("\n");
}
