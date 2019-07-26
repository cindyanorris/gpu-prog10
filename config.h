#define THREADSPERBLOCK 1024 
//The larger the value below gets, the smaller the number of threads
//per block could get for the sum kernel. The sum kernel threads per
//block is THREADPERBLOCK/numElements. You don't want that number
//to be less than 32.
#define MAXELES 32
#define MINVEC 8
#define MAXVEC 30
