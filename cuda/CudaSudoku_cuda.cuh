#ifndef CUDA_SUDOKU_CUDA_CUH
#define CUDA_SUDOKU_CUDA_CUH


#define N 9
#define n 3

/* TODO: This is a CUDA header file.
If you have any functions in your .cu file that need to be
accessed from the outside, declare them here */

void cudaSudokuBacktrack(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        int *boards,
        const int numBoards,
        int *finished,
        int *solved);



#endif // CUDA_SUDOKU_CUDA_CUH
