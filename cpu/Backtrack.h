#include <iostream>
#include <set>

using namespace std;

class Backtrack {
private:
    int *board;         // sudoku board
    int N;              // dimensions of board (normally 9)
    int n;              // dimensions of sub-board (normally 3)

    // Helper functions

    // return true if the current board is valid
    bool validBoard();
    // version where the only change on the board is the value at row r and col c
    bool validBoard(int r, int c);

    // determine if the board is done, which means no empty spots
    bool doneBoard();

    // print the board
    void printBoard();

    // find an empty spot on the board after location and set the input parameters to the row/col
    // value of the next empty spot
    // return true if an empty spot was found
    bool findEmptySpot(int &row, int &col);

    // helper for solving function to recursively backtrack in search of a valid sudoku board
    bool solveHelper();

public:
    // constructors and destructors
    Backtrack(int *board, int N, int n);
    ~Backtrack();

    // solve board
    void solve();
};
