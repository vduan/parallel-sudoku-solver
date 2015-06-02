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
bool validBoard(const int *board) {
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
bool validBoard(const int *board, int r, int c) {

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
bool doneBoard(const int *board) {
    for (int i = 0; i < N * N; i++) {
        if (board[i] == 0) {
            return false;
        }
    }

    return true;
}


__device__
bool findEmptySpot(const int *board, int *row, int *col) {
    printf("calling findEmptySpot\n");
    printf("board = %d and board[0][0] = %d\n", board, board[0 * N + 0]);
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            if (board[r * N + c] == 0) {
                printf("r = %d and c = %d\n", r, c);
                *row = r;
                *col = c;
                return true;
            }
        }
    }
    // for (*row = 0; *row < N; *row = *row + 1) {
    //     for (*col = 0; *col < N; *col = *col + 1) {
    //         if (board[*row * N + *col] == 0) {
    //             return true;
    //         }
    //     }
    // }

    return false;
}


__device__
bool solveHelper(int *board) {
    printf("calling solveHelper with board = %d\n", board);

    int row = 10;
    int col = 10;
    if (!findEmptySpot(board, &row, &col)) {
        return true;
    }

    printf("row = %d and col = %d and board[row, col] = %d\n", row, col, board[row * N + col]);

    for (int k = 1; k <= N; k++) {
        board[row * N + col] = k;
        if (validBoard(board, row, col)) {
            if (solveHelper(board)) {
                return true;
            }
        }
        // if (validBoard(board, row, col) && solveHelper(board)) {
        //     return true;
        // }
        board[row * N + col] = 0;
    }

    return false;
}


__device__
bool solve(int *board) {

    printf("starting solve\n");

    // initial board is invalid
    if (!validBoard(board, -1, -1)) {

        printf("solve: invalid board\n");
        return false;
    }

    // board is already solved
    if (doneBoard(board)) {

        printf("solve: done board\n");
        return true;
    }

    // otherwise, try to solve the board
    if (solveHelper(board)) {

        // solved
        printf("solve: solved board\n");
        return true;
    } else {

        // unsolvable
        printf("solve: unsolvable\n");
        return false;
    }
}


__global__
void sudokuBacktrack(int *boards, const int numBoards, int *finished, int *solved) {

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    int *currentBoard;

    // printf("done = %d and index = %d\n", *done, index);

    while ((*finished == 0) && (index < numBoards)) {

        printf("in kernel with index: %d and finished = %d\n", index, *finished);

        currentBoard = boards + index * 81;

        if (solve(currentBoard)) {
            printf("solved at index: %d\n", index);
            // solved
            *finished = 1;

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
        int *boards,
        const int numBoards,
        int *finished,
        int *solved) {

    printf("calling kernel\n");

    cudaDeviceSetLimit(cudaLimitStackSize, 1 * N * N * 10000000 * sizeof(float));


    sudokuBacktrack<<<blocks, threadsPerBlock>>>(boards, numBoards, finished, solved);
}
