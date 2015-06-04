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
  const unsigned int threadsPerBlock = atoi(argv[1]);
  const unsigned int maxBlocks = atoi(argv[2]);
  const unsigned int iterations = atoi(argv[3]);
  int *new_boards;
  int *old_boards;
  int *empty_spaces;
  int *empty_space_count;

  const int sk = 40000000; // = 5^9
  cudaMalloc(&empty_spaces, sk * sizeof(int));
  cudaMalloc(&empty_space_count, (sk / 81 + 1) * sizeof(int));
  cudaMalloc(&new_boards, sk * sizeof(int));
  cudaMalloc(&old_boards, sk * sizeof(int));
  // where to store the next new board generated
  int *board_index;
  // same as board index, except we need to set board_index to zero every time and this can stay
  int total_boards = 1;
  cudaMalloc(&board_index, sizeof(int));
  cudaMemset(board_index, 0, sizeof(int));
  cudaMemset(new_boards, 0, sk * sizeof(int));
  cudaMemset(old_boards, 0, sk * sizeof(int));
  int *sudokus = (int*) malloc(sk * sizeof(int));
  // the actual bound
  //memset(sudokus, 0, n* sizeof(int));
  int first[81] = {0,2,0,1,7,8,0,3,0,
                    0,4,0,3,0,2,0,9,0,
                    1,0,0,0,0,0,0,0,6,
                    0,0,8,6,0,3,5,0,0,
                    3,0,0,0,0,0,0,0,4,
                    0,0,6,7,0,9,2,0,0,
                    9,0,0,0,0,0,0,0,2,
                    0,8,0,9,0,1,0,6,0,
                    0,1,0,4,3,6,0,5,0};

  cudaMemcpy(old_boards, first, 81 * sizeof(int), cudaMemcpyHostToDevice);
  callBFSKernel(maxBlocks, threadsPerBlock, old_boards, new_boards, total_boards, board_index, empty_spaces, empty_space_count);

  int *host_count = (int*) malloc(sizeof(int));


  for (unsigned int i = 0; i < iterations; i++) {
      cudaMemcpy(host_count, board_index, sizeof(int), cudaMemcpyDeviceToHost);
      printf("total boards after an iteration %d: %d\n", i, *host_count);
           //cudaMemcpy(old_boards, new_boards, 81 * (*host_count) * sizeof(int), cudaMemcpyDeviceToDevice);
      cudaMemset(board_index, 0, sizeof(int));
      //cudaMemset(new_boards, 0, sk * sizeof(int)); 
      if (i % 2 == 0) {
        callBFSKernel(maxBlocks, threadsPerBlock, new_boards, old_boards, (*host_count), board_index, empty_spaces, empty_space_count);
      }
      else {
        callBFSKernel(maxBlocks, threadsPerBlock, old_boards, new_boards, (*host_count), board_index, empty_spaces, empty_space_count);
      }
  }

  cudaMemcpy(host_count, board_index, sizeof(int), cudaMemcpyDeviceToHost);
  printf("new number of boards retrieved is %d\n", *host_count);

  int *dev_finished;
  int *dev_solved;

  cudaMalloc(&dev_finished, sizeof(int));
  cudaMalloc(&dev_solved, N * N * sizeof(int));

  cudaMemset(dev_finished, 0, sizeof(int));
  cudaMemset(dev_solved, 0, N * N * sizeof(int));

  printf("before kernel call\n");
  if (iterations % 2 == 1) {
     // if odd number of iterations run, then send it old boards not new boards;
     new_boards = old_boards;
  }
  cudaSudokuBacktrack(maxBlocks, threadsPerBlock, new_boards, (*host_count), empty_spaces, empty_space_count, dev_finished, dev_solved);

  printf("after kernel call\n");

  int *solved = new int[N * N];

  memset(solved, 0, N * N * sizeof(int));

  cudaMemcpy(solved, dev_solved, N * N * sizeof(int), cudaMemcpyDeviceToHost);

  printBoard(solved);


  cudaFree(dev_finished);
  cudaFree(dev_solved);
  delete[] solved;
  cudaFree(board_index);
  cudaFree(new_boards);
  cudaFree(old_boards);
  free(sudokus);



  
  return 0;
}
