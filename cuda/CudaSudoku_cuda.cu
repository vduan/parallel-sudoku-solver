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

        //printf("in kernel with index = %d and finished = %d\n", index, *finished);
    
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

    //printf("calling kernel\n");

    sudokuBacktrack<<<blocks, threadsPerBlock>>>(boards, numBoards, emptySpaces, numEmptySpaces, finished, solved);
}

__global__
void
cudaBFSKernel(int *old_boards, int *new_boards, int total_boards, int *board_index, int *empty_spaces, int *empty_space_count) {
    
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // board_index must start at zero 

    while (index < total_boards) {
        //printf("index is %d\n", index);
        // find the next empty spot
        int found = 0;
        for (int i = (index * 81); (i < (index * 81) + 81) && (found == 0); i++) {
            //printf("checking index number %d, number there is %d\n", i, old_boards[i]);
            // found a open spot
            if (old_boards[i] == 0) {
                found = 1;
                // get the correct row and column shits
                //printf("open spot found is %d\n", i);
                int temp = i - 81 * index;
                int row = temp / 9;
                int col = temp % 9;
                
                
                //printf("row and column of open spot is %d, %d\n", row, col);
                // figure out which numbers work here
                for (int attempt = 1; attempt <= 9; attempt++) {
                    //printf("trying the number: %d\n", attempt);
                    int works = 1;
                    // row constraint, test various columns
                    for (int c = 0; c < 9; c++) {
                        if (old_boards[row * 9 + c + 81 * index] == attempt) {
                            //printf("attempt number %d breaks here M\n", attempt);
                            works = 0;
                        }
                    }
                    // column contraint, test various rows
                    for (int r = 0; r < 9; r++) {
                        if (old_boards[r * 9 + col + 81 * index] == attempt) {
                            //printf("attempt number %d breaks here J\n", attempt);
                            works = 0;
                        }
                    }
                    // box constraint
                    for (int r = 3 * (row / 3); r < 3; r++) {
                        for (int c = 3 * (col / 3); c < 3; c++) {
                            if (old_boards[r * 9 + c + 81 * index] == attempt) {
                            //printf("attempt number %d breaks here T for row, col = (%d,%d)\n", attempt, r, c);
                                works = 0;
                            }
                        }
                    }
                    if (works == 1) {
                        //printf("the number: %d works\n", attempt);
                        // copy the whole board

                        int next_board_index = atomicAdd(board_index, 1);
                        int empty_index = 0;
                        for (int r = 0; r < 9; r++) {
                            for (int c = 0; c < 9; c++) {
                                new_boards[next_board_index * 81 + r * 9 + c] = old_boards[index * 81 + r * 9 + c];
                                if (old_boards[index * 81 + r * 9 + c] == 0 && (r != row || c != col)) {
                                    empty_spaces[empty_index + 81 * next_board_index] = r * 9 + c;

                                    empty_index++;
                                    //printf("row and column of open spot is %d, %d\n", r, c);
                                    //printf("empty spots: %d\n", empty_index);
                                }
                            }
                        }
                        empty_space_count[next_board_index] = empty_index;
                        //printf("the skleeg is : %d\n", empty_index);
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

