#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>

#include "CudaSudoku_cuda.cuh"


__device__
void clearBitmap(bool *map, int size) {
    for (int i = 0; i < size; i++) {
        map[i] = false;
    }
}

__device__
bool validBoard(float *board) {
    bool seen[N];
    clearBitmap(seen, N);

    // check if rows are valid
    for (int i = 0; i < N; i++) {
        clearBitmap(seen, N);

        for (int j = 0; j < N; j++) {
            int val = board[i * N + j];

            if (val != 0) {
                if (seen[val - 1]) {
                    return false;
                } else {
                    seen[val - 1] = true;
                }
            }
        }
    }

    // check if columns are valid
    for (int j = 0; j < N; j++) {
        clearBitmap(seen, N);

        for (int i = 0; i < N; i++) {
            int val = board[i * N + j];

            if (val != 0) {
                if (seen[val - 1]) {
                    return false;
                } else {
                    seen[val - 1] = true;
                }
            }
        }
    }

    // finally check if the sub-boards are valid
    for (int ridx = 0; ridx < n; ridx++) {
        for (int cidx = 0; cidx < n; cidx++) {
            clearBitmap(seen, N);

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int val = board[(ridx * n + i) * N + (cidx * n + j)];

                    if (val != 0) {
                        if (seen[val - 1]) {
                            return false;
                        } else {
                            seen[val-1] = true;
                        }
                    }
                }
            }
        }
    }


    // if we get here, then the board is valid
    return true;
}


__device__
bool validBoard(float *board, int r, int c) {

    // if r is less than 0, then just default case
    if (r < 0) {
        return validBoard(board);
    }

    bool seen[N];
    clearBitmap(seen, N);

    // check if row is valid
    for (int i = 0; i < N; i++) {
        int val = board[r * N + i];

        if (val != 0) {
            if (seen[val - 1]) {
                return false;
            } else {
                seen[val - 1] = true;
            }
        }
    }

    // check if column is valid
    clearBitmap(seen, N);
    for (int j = 0; j < N; j++) {
        int val = board[j * N + c];

        if (val != 0) {
            if (seen[val - 1]) {
                return false;
            } else {
                seen[val - 1] = true;
            }
        }
    }

    // finally check if the sub-board is valid
    int ridx = r / n;
    int cidx = c / n;

    clearBitmap(seen, N);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int val = board[(ridx * n + i) * N + (cidx * n + j)];

            if (val != 0) {
                if (seen[val - 1]) {
                    return false;
                } else {
                    seen[val - 1] = true;
                }
            }
        }
    }

    // if we get here, then the board is valid
    return true;
}


__device__
bool doneBoard(float *board) {
    for (int i = 0; i < N * N; i++) {
        if (board[i] == 0) {
            return false;
        }
    }

    return true;
}


__device__
bool findEmptySpot(float *board, int *row, int *col) {
    for (*row = 0; *row < N; *row++) {
        for (*col = 0; *col < N; *col++) {
            if (board[*row * N + *col] == 0) {
                return true;
            }
        }
    }

    return false;
}


__device__
bool solveHelper(float *board) {

    int row, col;
    if (!findEmptySpot(board, &row, &col)) {
        return true;
    }

    for (int k = 1; k <= N; k++) {
        board[row * N + col] = k;
        if (validBoard(board, row, col) && solveHelper(board)) {
            return true;
        }
        board[row * N + col] = 0;
    }

    return false;
}


__device__
bool solve(float *board) {

    // initial board is invalid
    if (!validBoard(board, -1, -1)) {
        return false;
    }

    // board is already solved
    if (doneBoard(board)) {
        return true;
    }

    // otherwise, try to solve the board
    if (solveHelper(board)) {

        // solved
        return true;
    } else {

        // unsolvable
        return false;
    }
}


__global__
void sudokuBacktrack(float *boards, const int numBoards, int *done, float *solved) {

    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    float *currentBoard;

    while (!done && index < numBoards) {

        currentBoard = boards + index * 81;

        if (solve(currentBoard)) {
            // solved
            *done = 1;

            // return the solved board
            for (int i = 0; i < N * N; i++) {
                solved[i] = currentBoard[i];
            }
        }



        index += gridDim.x * blockDim.x;
    }
}



void cudaSudokuBacktrack(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        float *boards,
        const int numBoards,
        int *done,
        float *solved) {
    sudokuBacktrack<<<blocks, threadsPerBlock>>>(boards, numBoards, done, solved);
}
