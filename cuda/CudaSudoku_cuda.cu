#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>

#include "CudaSudoku_cuda.cuh"


/**
 * This function takes in a bitmap and clears them all to false.
 */
__device__
void clearBitmap(bool *map, int size) {
    for (int i = 0; i < size; i++) {
        map[i] = false;
    }
}


/**
 * This device checks the entire board to see if it is valid.
 *
 * board: this is a N * N sized array that stores the board to check. Rows are stored contiguously,
 *        so to access row r and col c, use board[r * N + c]
 */
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


/**
 * This function takes a board and an index between 0 and N * N - 1. This function assumes the board
 * without the value at changed is valid and checks for validity given the new change.
 *
 * board:   this is a N * N sized array that stores the board to check. Rows are stored
 *          contiguously, so to access row r and col c, use board[r * N + c]
 *
 * changed: this is an integer that stores the index of the board that was changed
 */
__device__
bool validBoard(const int *board, int changed) {

    int r = changed / 9;
    int c = changed % 9;

    // if changed is less than 0, then just default case
    if (changed < 0) {
        return validBoard(board);
    }

    if ((board[changed] < 1) || (board[changed] > 9)) {
        return false;
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


/**
 * This kernel has each thread try to solve a different board in the input array using the
 * backtracking algorithm.
 *
 * boards:      This is an array of size numBoards * N * N. Each board is stored contiguously,
 *              and rows are contiguous within the board. So, to access board x, row r, and col c,
 *              use boards[x * N * N + r * N + c]
 *
 * numBoards:   The total number of boards in the boards array.
 *
 * emptySpaces: This is an array of size numBoards * N * N. board is stored contiguously, and stores
 *              the indices of the empty spaces in that board. Note that this N * N pieces may not
 *              be filled.
 *
 * numEmptySpaces:  This is an array of size numBoards. Each value stores the number of empty spaces
 *                  in the corresponding board.
 *
 * finished:    This is a flag that determines if a solution has been found. This is a stopping
 *              condition for the kernel.
 *
 * solved:      This is an output array of size N * N where the solved board is stored.
 */
__global__
void sudokuBacktrack(int *boards,
        const int numBoards,
        int *emptySpaces,
        int *numEmptySpaces,
        int *finished,
        int *solved) {

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    int *currentBoard;
    int *currentEmptySpaces;
    int currentNumEmptySpaces;


    while ((*finished == 0) && (index < numBoards)) {
    
        int emptyIndex = 0;

        currentBoard = boards + index * 81;
        currentEmptySpaces = emptySpaces + index * 81;
        currentNumEmptySpaces = numEmptySpaces[index];

        while ((emptyIndex >= 0) && (emptyIndex < currentNumEmptySpaces)) {

            currentBoard[currentEmptySpaces[emptyIndex]]++;

            if (!validBoard(currentBoard, currentEmptySpaces[emptyIndex])) {

                // if the board is invalid and we tried all numbers here already, backtrack
                // otherwise continue (it will just try the next number in the next iteration)
                if (currentBoard[currentEmptySpaces[emptyIndex]] >= 9) {
                    currentBoard[currentEmptySpaces[emptyIndex]] = 0;
                    emptyIndex--;
                }
            }
            // if valid board, move forward in algorithm
            else {
                emptyIndex++;
            }

        }

        if (emptyIndex == currentNumEmptySpaces) {
            // solved board found
            *finished = 1;

            // copy board to output
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
        int *emptySpaces,
        int *numEmptySpaces,
        int *finished,
        int *solved) {

    sudokuBacktrack<<<blocks, threadsPerBlock>>>
        (boards, numBoards, emptySpaces, numEmptySpaces, finished, solved);
}


/**
 * This kernel takes a set of old boards and finds all possible next boards by filling in the next
 * empty space.
 *
 * old_boards:      This is an array of size sk. Each N * N section is another board. The rows
 *                  are contiguous within the board. This array stores the previous set of boards.
 *
 * new_boards:      This is an array of size sk. Each N * N section is another board. The rows
 *                  are contiguous within the board. This array stores the next set of boards.
 *
 * total_boards:    Number of old boards.
 *
 * board_index:     Index specifying the index of the next opening in new_boards.
 *
 * empty_spaces:    This is an array of size sk. Each N * N section is another board, storing the
 *                  indices of empty spaces in new_boards.
 *
 * empty_space_count:   This is an array of size sk / N / N + 1 which stores the number of empty
 *                      spaces in the corresponding board.
 */
__global__
void
cudaBFSKernel(int *old_boards,
        int *new_boards,
        int total_boards,
        int *board_index,
        int *empty_spaces,
        int *empty_space_count) {
    
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // board_index must start at zero 

    while (index < total_boards) {
        // find the next empty spot
        int found = 0;

        for (int i = (index * N * N); (i < (index * N * N) + N * N) && (found == 0); i++) {
            // found a open spot
            if (old_boards[i] == 0) {
                found = 1;
                // get the correct row and column shits
                int temp = i - N * N * index;
                int row = temp / N;
                int col = temp % N;
                
                // figure out which numbers work here
                for (int attempt = 1; attempt <= N; attempt++) {
                    int works = 1;
                    // row constraint, test various columns
                    for (int c = 0; c < N; c++) {
                        if (old_boards[row * N + c + N * N * index] == attempt) {
                            works = 0;
                        }
                    }
                    // column contraint, test various rows
                    for (int r = 0; r < N; r++) {
                        if (old_boards[r * N + col + N * N * index] == attempt) {
                            works = 0;
                        }
                    }
                    // box constraint
                    for (int r = n * (row / n); r < n; r++) {
                        for (int c = n * (col / n); c < n; c++) {
                            if (old_boards[r * N + c + N * N * index] == attempt) {
                                works = 0;
                            }
                        }
                    }
                    if (works == 1) {
                        // copy the whole board

                        int next_board_index = atomicAdd(board_index, 1);
                        int empty_index = 0;
                        for (int r = 0; r < 9; r++) {
                            for (int c = 0; c < 9; c++) {
                                new_boards[next_board_index * 81 + r * 9 + c] = old_boards[index * 81 + r * 9 + c];
                                if (old_boards[index * 81 + r * 9 + c] == 0 && (r != row || c != col)) {
                                    empty_spaces[empty_index + 81 * next_board_index] = r * 9 + c;

                                    empty_index++;
                                }
                            }
                        }
                        empty_space_count[next_board_index] = empty_index;
                        new_boards[next_board_index * 81 + row * 9 + col] = attempt;
                    }
                }
            }
        }

        index += blockDim.x * gridDim.x;
    }
}


void callBFSKernel(const unsigned int blocks, 
                        const unsigned int threadsPerBlock,
                        int *old_boards,
                        int *new_boards,
                        int total_boards,
                        int *board_index,
                        int *empty_spaces,
                        int *empty_space_count) {
    cudaBFSKernel<<<blocks, threadsPerBlock>>>
        (old_boards, new_boards, total_boards, board_index, empty_spaces, empty_space_count);
}

