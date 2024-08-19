/**
*   CS6023: GPU Programming 
*   Assignment 2
*   
*   Please don't change any existing code in this file.
*
*   Please add necessary memory APIs for your implementation. Use cudaFree() 
*   to free up memory as soon as you're done with an allocation. 
*   This will ensure that you don't run out of memory while running
*   large test cases. Use the minimum required memory for your 
*   implementation. DO NOT change the kernel configuration parameters.
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

__global__ void fun(long int *inputMatrix,long int *filterM,long int *outputMatrix, int m, int n, int k){
    int i = blockIdx.x;
    int j = threadIdx.x;
    extern __shared__ long int filterShared[];
    long int *filter = filterShared;
    if(threadIdx.x == 0){
      for(int i = 0; i < k*k; i++) filter[i] = filterM[i];
    }
    __syncthreads();
    long int sum = 0;
    for (int p = 0; p < k; ++p) {
        for (int q = 0; q < k; ++q) {
            int inputIndex = ((i + p - k/2) * n) + (j + q - k/2);
            int filterIndex = (p * k) + q;
            if (i + p - k/2 >= 0 && i + p - k/2 < m && j + q - k/2 >= 0 && j + q - k/2 < n) {
                sum += inputMatrix[inputIndex] * filter[filterIndex];
            }
        }
    }
    outputMatrix[i*n + j] = sum;
}

int main(int argc, char** argv) {

    int m,n,k;
    cin>>m>>n>>k;


    long int* h_mat = new long int[m * n];
    long int* h_filter = new long int[k * k];

    long int* h_ans = new long int[m * n];


    for (long int i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }

    for (long int i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }

    /**
     * 
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     * 
    **/

    /****************************************************Start Here***********************************************************/
    
    long int *d_input, *d_filter, *d_output;

    cudaMalloc(&d_input, m * n * sizeof(long int));
    cudaMalloc(&d_filter, k * k * sizeof(long int));
    cudaMalloc(&d_output, m * n * sizeof(long int));

    cudaMemcpy(d_input, h_mat, m * n * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, k * k * sizeof(long int), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();//keep it just before the kernel launch
    fun<<<m, n,k*k*sizeof(long int)>>>(d_input, d_filter, d_output, m, n, k);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();//keep it just after the kernel launch
    
    cudaMemcpy(h_ans, d_output, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
    
    
    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     * 
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     * 
    */
    
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}