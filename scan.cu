#include <stdio.h>
#include <stdlib.h>
#include "h_scan.h"
#include "d_scan.h"
#include "wrappers.h"
//config.h defines the number of threads in a block (THREADSPERBLOCK), 
//the minimum value used for calculating the vector size (MINVEC),
//and the maximum value used for calculating the vector size (MAXVEC).
#include "config.h"     

//prototypes for functions in this file
static void initVector(int * array, int length);
static void parseArgs(int argc, char * argv[], int *, int *);
static void compare(int * result1, int * result2, int n);
static void printUsage();
static int isPowerOfTwo(unsigned int n);
void printVector(int * vector, int vectorLen);

/*
   driver for the exclusive scan program.  
*/
int main(int argc, char * argv[])
{
    int vectorLen, numEles;
    //get the length of the vector and the number of elements
    //partitioned to threads to finish the GPU scan
    parseArgs(argc, argv, &vectorLen, &numEles);
    int * h_vector = (int *) Malloc(sizeof(int) * vectorLen);
    int * h_result = (int *) Malloc(sizeof(int) * vectorLen);
    int * d_result = (int *) Malloc(sizeof(int) * vectorLen);
    float h_time, d_time, speedup;

    //initialize vector 
    initVector(h_vector, vectorLen);
    //printVector(h_vector, vectorLen);
  
    printf("\nScan of vector of size: %d\n", vectorLen);
    printf("Number of sums computed by a thread: %d\n", numEles); 

    //perform the scan on the CPU
    h_time = h_scan(h_result, h_vector, vectorLen);
    //printVector(h_result, vectorLen);
    printf("\nTiming\n");
    printf("------\n");
    printf("CPU: \t\t%f msec\n", h_time);

    //perform the scan on the GPU 
    d_time = d_scan(d_result, h_vector, vectorLen, numEles);
    //printVector(d_result, vectorLen);

    //compare GPU and CPU results 
    compare(h_result, d_result, vectorLen);
    printf("GPU: \t\t%f msec\n", d_time);
    speedup = h_time/d_time;
    printf("Speedup: \t%f\n", speedup);

    free(h_result);
    free(d_result);
    free(h_vector);
}    

/* 
    parseArgs
    This function parses the command line arguments to get
    the vector length of the vector for the scan.
    It also sets the number of elements allocated 
    to a single thread to complete the final scan step. If 
    the vector length or number of elements value is invalid, 
    it prints usage information and exits.
    Inputs:
    argc - count of the number of command line arguments
    argv - array of command line arguments
    vectorLen - pointer to an int to be set to the vector length
    numEles - pointer to an int to be set to the number of elements
              that the GPU code will partition to each thread 
              when completing the final sum of the exclusive scan
*/
void parseArgs(int argc, char * argv[], int * vectorLen, int * numEles)
{
    int vlen = (1 << MINVEC);  //set vector length to default
    int vfactor;
    int i, numeles = 1; 
    for (i = 1; i < argc; i++)
    {
        if (strncmp(argv[i], "-s", 3) == 0)
        {
            //get value provided by user to calculate the vector size
            vfactor = atoi(argv[i+1]);
            if (vfactor < MINVEC || vfactor > MAXVEC)
            {
                printf("\nInvalid vector size: %d\n\n", vfactor);
                printUsage();
            }
            //calculate the vector size
            vlen = (1 << vfactor);
            i++;
        } else if (strncmp(argv[i], "-n", 3) == 0)
        {
            //get value provided by user for partitioning on GPU
            numeles = atoi(argv[i+1]);
            if (!isPowerOfTwo(numeles) || numeles > MAXELES)
            {
                printf("\nInvalid partition size: %d\n\n", numeles);
                printUsage();
            }
            i++;
        } else if (strncmp(argv[i], "-h", 3) == 0)
        {
            //display help info
            printUsage();
        } else
        {
            printf("\nInvalid option %s\n\n", argv[i]);
            printUsage();
        }
    }
    (*vectorLen) = vlen;
    (*numEles) = numeles;
}

/*
    printUsage
    prints usage information and exits
*/
void printUsage()
{
    printf("\nThis program performs an exclusive scan of a vector.\n"); 
    printf("The scan is performed on the CPU and the GPU. The program\n");
    printf("verifies the GPU results by comparing them to the CPU\n");
    printf("results and outputs the times it takes to perform each scan.\n");
    printf("usage: scan [-h] [-s <vector size>] [-n <partition size>]\n");
    printf("       [-h] print usage information\n");
    printf("       <vector size> size of randomly generated vector");
    printf(" is (1 << <vector size>).\n");
    printf("                   Min <vector size> is %d.\n", MINVEC);            
    printf("                   Max <vector size> is %d.\n", MAXVEC);            
    printf("                   Default is %d.\n", (1 << MINVEC));
    printf("       <partition size> number of sums computed by a thread");
    printf(" when doing the\n");
    printf("                   final sum to complete the scan.\n");
    printf("                   Must be a power of 2 that is less than");
    printf(" or equal to %d.\n", MAXELES);
    printf("                   Default is 1.\n");
    exit(EXIT_FAILURE);
}

/* 
    initVector
    Initializes an array of ints of size
    length to random values between 0 and 5. 
    Inputs:
    array - pointer to the array to initialize
    length - length of array
*/
void initVector(int * array, int length)
{
    int i;
    for (i = 0; i < length; i++)
    {
        array[i] = (rand() % 5);
    }
}

/*
    compare
    Compares the values in two vectors and outputs an
    error message and exits if the values do not match.
    result1, result2 - int vectors
    n - length of each vector
*/
void compare(int * result1, int * result2, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        int diff = abs(result1[i] - result2[i]);
        if (diff != 0) 
        {
            printf("GPU scan does not match CPU scan.\n");
            printf("cpu result[%d]: %d, gpu: result[%d]: %d\n", 
                   i, result1[i], i, result2[i]);
            exit(EXIT_FAILURE);
        }
    }
}

/*
    printVector
    prints the contents of a vector, 10 elements per line
    vector - pointer to the vector
    vectorLen - length of vector
*/
void printVector(int * vector, int vectorLen)
{
    for (int i = 0; i < vectorLen; i++)
    {
        if ((i % 10) == 0)printf("\n%4d: ", i);
        printf("%3d ", vector[i]);
    }
    printf("\n");
}

/*
   isPowerOfTwo
   returns true if parameter is a power of 2 > 0.
*/ 
int isPowerOfTwo(unsigned int n)
{
    return (n != 0) && ((n & (n - 1)) == 0);
}
