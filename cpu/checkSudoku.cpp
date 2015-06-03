#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

#include "Backtrack.h"

#define N 9
#define n 3

using namespace std;



void clearBitmap(bool *map, int size) {
    for (int i = 0; i < size; i++) {
        map[i] = false;
    }
}


bool validBoard(int *board) {
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



bool validBoard(int *board, int r, int c) {

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



bool doneBoard(int *board) {
    for (int i = 0; i < N * N; i++) {
        if (board[i] == 0) {
            return false;
        }
    }

    return true;
}



bool findEmptySpot(int *board, int *row, int *col) {
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            if (board[r * N + c] == 0) {
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



bool solveHelper(int *board) {
    int row = 10;
    int col = 10;
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



bool solve(int *board) {

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


// main driver to test the sudoku program
int main() {
    // create a sudoku board
    // int n = 3;          // dimensions of sub-board
    // int N = n * n;      // dimensions of board

    int *board = new int[N * N];
    load("puzzle.txt", board);

    if (solve(board)) {
        // solved
        cout << "solved" << endl;

        // return the solved board
        printBoard(board);
    }


    return 0;
}
