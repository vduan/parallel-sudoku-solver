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

  int *emptySpaces = new int[N * N];
  memset(emptySpaces, 0, N * N * sizeof(int));
  int numEmptySpaces = 0;
  for (int i = 0; i < N * N; i++) {
    if (board[i] == 0) {
      emptySpaces[numEmptySpaces++] = i;
    }
  }

  int numBoards = 1;

  int *dev_boards;
  int *dev_finished;
  int *dev_emptySpaces;
  int *dev_numEmptySpaces;
  int *dev_solved;

  cudaMalloc(&dev_boards, numBoards * N * N * sizeof(int));
  cudaMalloc(&dev_finished, sizeof(int));
  cudaMalloc(&dev_emptySpaces, numBoards * N * N * sizeof(int));
  cudaMalloc(&dev_numEmptySpaces, numBoards * sizeof(int));
  cudaMalloc(&dev_solved, N * N * sizeof(int));

  cudaMemcpy(dev_boards, board, numBoards * N * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(dev_finished, 0, sizeof(int));
  cudaMemcpy(dev_emptySpaces, emptySpaces, numBoards * N * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_numEmptySpaces, &numEmptySpaces, numBoards * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(dev_solved, 0, N * N * sizeof(int));

  printf("before kernel call\n");

  cudaSudokuBacktrack(maxBlocks, threadsPerBlock, dev_boards, numBoards, dev_emptySpaces, dev_numEmptySpaces, dev_finished, dev_solved);

  printf("after kernel call\n");

  int *solved = new int[N * N];

  memset(solved, 0, N * N * sizeof(int));

  cudaMemcpy(solved, dev_solved, N * N * sizeof(int), cudaMemcpyDeviceToHost);

  printBoard(solved);


  cudaFree(dev_boards);
  cudaFree(dev_finished);
  cudaFree(dev_solved);
  delete[] board;
  delete[] emptySpaces;
  delete[] solved;
  
  return 0;
}
