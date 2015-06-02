#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>


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

  return 0;

  
  // const unsigned int blocks = std::min(maxBlocks, (unsigned int) ceil(
  //             numberOfNodes/float(threadsPerBlock)));

  float *dev_boards;
  int *dev_output;
  int *done;
  float *dev_solved;

  cudaMalloc(&dev_boards, sizeof(float));
  cudaMalloc(&dev_output, sizeof(int));
  cudaMalloc(&done, sizeof(int));
  cudaMalloc(&dev_solved, N * N * sizeof(float));

  cudaMemset(done, 0, sizeof(int));

  cudaSudokuBacktrack(maxBlocks, threadsPerBlock, dev_boards, 10, done, dev_solved);

  int *output = (int*) malloc(sizeof(int));

  cudaMemcpy(output, dev_output, sizeof(int), cudaMemcpyDeviceToHost);

  printf("output: %d\n", *output);
  
  return 0;
}
