#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstring>


#include <cuda_runtime.h>
#include <algorithm>
#include <curand.h>

#include "CudaSudoku_cuda.cuh"



void load(char *FileName, int *board) {
    FILE * a_file = fopen(FileName, "r");

    if (a_file == NULL) {
      printf("File load fail!\n"); return;
    }

    char temp;

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        if (!fscanf(a_file, "%c\n", &temp)) {
          printf("File loading error!\n");
          return;
        }

        if (temp >= '1' && temp <= '9') {
          board[i * N + j] = (int) (temp - '0');
        } else {
          board[i * N + j] = 0;
        }
      }
    }
}


void printBoard(int *board) {
  for (int i = 0; i < N; i++) {
    if (i % n == 0) {
      printf("-----------------------\n");
    }

    for (int j = 0; j < N; j++) {
      if (j % n == 0) {
        printf("| ");
      }
      printf("%d ", board[i * N + j]);
    }

    printf("|\n");
  }
  printf("-----------------------\n");
}





int main(int argc, char* argv[]) {
    
    
  if (argc < 4){
      printf("Usage: (threads per block) (max number of blocks) (filename)\n");
      exit(-1);
  }

  
  /* Additional parameters for the assignment */
  const unsigned int threadsPerBlock = atoi(argv[1]);
  const unsigned int maxBlocks = atoi(argv[2]);
  char *filename = argv[3];

  int *board = new int[N * N];
  load(filename, board);

  printBoard(board);
  
  // const unsigned int blocks = std::min(maxBlocks, (unsigned int) ceil(
  //             numberOfNodes/float(threadsPerBlock)));

  cudaDeviceSetLimit(cudaLimitStackSize, 1 * N * N * 10000000 * sizeof(float));

  int *dev_boards;
  int *dev_finished;
  int *dev_solved;

  cudaMalloc(&dev_boards, N * N * sizeof(int));
  cudaMalloc(&dev_finished, sizeof(int));
  cudaMalloc(&dev_solved, N * N * sizeof(int));

  cudaMemcpy(dev_boards, board, N * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(dev_finished, 0, sizeof(int));
  cudaMemset(dev_solved, 0, N * N * sizeof(int));

  printf("before kernel call\n");

  cudaSudokuBacktrack(maxBlocks, threadsPerBlock, dev_boards, 1, dev_finished, dev_solved);

  printf("after kernel call\n");

  int *solved = new int[N * N];

  memset(solved, 0, N * N * sizeof(int));

  cudaMemcpy(solved, dev_solved, N * N * sizeof(int), cudaMemcpyDeviceToHost);

  printBoard(solved);


  cudaFree(dev_boards);
  cudaFree(dev_finished);
  cudaFree(dev_solved);
  delete[] board;
  delete[] solved;
  
  return 0;
}
