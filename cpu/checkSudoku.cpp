#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

#include "Backtrack.h"

using namespace std;


void initializeBoard(int *board, int N) {
    for (int i = 0; i < N * N; i++) {
        board[i] = 0;
    }

    return;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            board[i * N + j] = (3 * i + j + i / 3) % 9 + 1;
        }
    }

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, N * N - 1);

    for (int i = 0; i < 50; i++) {
        int idx = distribution(generator);

        board[idx] = 0;
    }
}


// initialize the board
void initializeBoard2(int *board, int N) {
        // initialize board to 0, which means empty spot
    for (int i = 0; i < N * N; i++) {
        board[i] = 0;
    }

    // set up a board, where 1 row is contiguous
    // so, row i and column j can be accessed at board[j + i * N];
    /*
     *  -------------------------
     *  | 0 0 3 | 0 7 0 | 0 6 0 |
     *  | 1 4 0 | 0 0 0 | 8 0 0 |
     *  | 6 5 0 | 0 0 4 | 9 0 0 |
     *  -------------------------
     *  | 7 6 0 | 8 0 2 | 0 0 9 |
     *  | 3 1 0 | 0 5 0 | 0 8 2 |
     *  | 2 0 0 | 9 0 3 | 0 1 5 |
     *  -------------------------
     *  | 0 0 1 | 7 0 0 | 0 9 8 |
     *  | 0 0 6 | 0 0 0 | 0 3 1 |
     *  | 0 9 0 | 0 8 0 | 5 0 0 |
     *  -------------------------
     */

    board[0 * N + 2] = 3;
    board[0 * N + 4] = 7;
    board[0 * N + 7] = 6;
    board[1 * N + 0] = 1;
    board[1 * N + 1] = 4;
    board[1 * N + 6] = 8;
    board[2 * N + 0] = 6;
    board[2 * N + 1] = 5;
    board[2 * N + 5] = 4;
    board[2 * N + 6] = 9;

    board[3 * N + 0] = 7;
    board[3 * N + 1] = 6;
    board[3 * N + 3] = 8;
    board[3 * N + 5] = 2;
    board[3 * N + 8] = 9;
    board[4 * N + 0] = 3;
    board[4 * N + 1] = 1;
    board[4 * N + 4] = 5;
    board[4 * N + 7] = 8;
    board[4 * N + 8] = 2;
    board[5 * N + 0] = 2;
    board[5 * N + 3] = 9;
    board[5 * N + 5] = 3;
    board[5 * N + 7] = 1;
    board[5 * N + 8] = 5;

    board[6 * N + 2] = 1;
    board[6 * N + 3] = 7;
    board[6 * N + 7] = 9;
    board[6 * N + 8] = 8;
    board[7 * N + 2] = 6;
    board[7 * N + 7] = 3;
    board[7 * N + 8] = 1;
    board[8 * N + 1] = 9;
    board[8 * N + 4] = 8;
    board[8 * N + 6] = 5;


    // old hard board
    // board[0 * N + 4] = 5;
    // board[0 * N + 5] = 7;
    // board[0 * N + 6] = 6;
    // board[0 * N + 7] = 4;

    // board[1 * N + 3] = 2;

    // board[2 * N + 0] = 7;
    // board[2 * N + 7] = 2;
    // board[2 * N + 8] = 3;

    // board[3 * N + 1] = 9;
    // board[3 * N + 4] = 6;
    // board[3 * N + 6] = 4;
    // board[3 * N + 7] = 5;

    // board[4 * N + 0] = 1;
    // board[4 * N + 8] = 7;

    // board[5 * N + 1] = 5;
    // board[5 * N + 2] = 6;
    // board[5 * N + 4] = 7;
    // board[5 * N + 7] = 3;

    // board[6 * N + 0] = 6;
    // board[6 * N + 1] = 1;
    // board[6 * N + 8] = 8;

    // board[7 * N + 5] = 6;

    // board[8 * N + 1] = 8;
    // board[8 * N + 2] = 3;
    // board[8 * N + 3] = 1;
    // board[8 * N + 4] = 2;
}

// print board
void printBoard(int *board, int N, int n) {
    for (int i = 0; i < n; i++) {
        cout << "-------------------------" << endl;
        for (int idx = i * n; idx < (i + 1) * n; idx++) {    
            for (int j = 0; j < 3; j++) {
                cout << "| ";
                for (int k = 0; k < n; k++) {
                    cout << board[idx * N + 3 * j + k] << " ";
                }
            }
            cout << "|" << endl;
        }
    }
}


// main driver to test the sudoku program
int main() {
    // create a sudoku board
    int n = 3;          // dimensions of sub-board
    int N = n * n;      // dimensions of board

    int board[N * N];

    initializeBoard(board, N);

    printBoard(board, N, n);

    // solve the board
    cout << "entering solver" << endl;
    Backtrack solver(board, N, n);

    solver.solve();

    cout << "done with solver" << endl;


    return 0;
}
