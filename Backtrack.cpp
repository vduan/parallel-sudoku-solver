#include "Backtrack.h"


bool Backtrack::validBoard() {
    set<int> seen;
    pair< set<int>::iterator, bool> ret;

    // check if rows are valid
    for (int i = 0; i < N; i++) {
        seen.clear();

        for (int j = 0; j < N; j++) {
            int val = board[i * N + j];
            ret = seen.insert(val);

            if (val != 0 && ret.second == false) {
                cout << "validBoard: row fail" << endl;
                return false;
            }
        }
    }

    // check if columns are valid
    for (int j = 0; j < N; j++) {
        seen.clear();

        for (int i = 0; i < N; i++) {
            int val = board[i * N + j];
            ret = seen.insert(val);

            if (val != 0 && ret.second == false) {
                cout << "validBoard: col fail" << endl;
                return false;
            }
        }
    }

    // finally check if the sub-boards are valid
    for (int ridx = 0; ridx < n; ridx++) {
        for (int cidx = 0; cidx < n; cidx++) {
            seen.clear();
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int val = board[(ridx * n + i) * N + (cidx * n + j)];
                    ret = seen.insert(val);

                    if (val != 0 && ret.second == false) {
                        cout << "validBoard: box fail" << endl;
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

void printSet(set<int> s) {
    cout << "\nprinting set\n";
    for (set<int>::iterator it = s.begin(); it != s.end(); it++) {
        cout << *it << "\t";
    }
    cout << endl << endl;
}


bool Backtrack::validBoard(int r, int c) {

    if (r < 0) {
        return Backtrack::validBoard();
    }


    set<int> seen;
    pair< set<int>::iterator, bool> ret;

    // first check if the row is valid
    for (int i = 0; i < N; i++) {
        int val = board[r * N + i];
        ret = seen.insert(val);

        if (val != 0 && ret.second == false) {
            return false;
        }
    }

    // next check if column is valid
    seen.clear();
    for (int i = 0; i < N; i++) {
        int val = board[i * N + c];
        ret = seen.insert(val);

        if (val != 0 && ret.second == false) {
            return false;
        }
    }

    // finally, check if the sub-board is valid
    // get sub-board indices. eg: middle square of standard board is at indices (1, 1)
    int ridx = r / n;
    int cidx = c / n;

    seen.clear();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int val = board[(ridx * n + i) * N + (cidx * n + j)];
            ret = seen.insert(val);

            if (val != 0 && ret.second == false) {
                return false;
            }
        }
    }

    // if code gets here, then board is valid
    return true;
}


bool Backtrack::doneBoard() {
    for (int i = 0; i < N * N; i++) {
        if (board[i] == 0) {
            return false;
        }
    }

    return true;
}


void Backtrack::printBoard() {
    for (int i = 0; i < n; i++) {
        cout << "-------------------------" << endl;
        for (int idx = i * n; idx < (i + 1) * n; idx++) {    
            for (int j = 0; j < n; j++) {
                cout << "| ";
                for (int k = 0; k < n; k++) {
                    cout << board[idx * N + n * j + k] << " ";
                }
            }
            cout << "|" << endl;
        }
    }
}


bool Backtrack::findEmptySpot(int &row, int &col) {
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            if (board[row * N + col] == 0) {
                return true;
            }
        }
    }

    return false;
}


bool Backtrack::solveHelper() {

    int row, col;

    if (!Backtrack::findEmptySpot(row, col)) {
        return true;
    }

    for (int k = 1; k <= N; k++) {
        board[row * N + col] = k;
        if (validBoard(row, col) && Backtrack::solveHelper()) {
            return true;
        }
        board[row * N + col] = 0;
    }

    return false;
}


Backtrack::Backtrack(int *board, int N, int n)
    : board(board)
    , N(N)
    , n(n)
{}


Backtrack::~Backtrack() {
}


void Backtrack::solve() {
    if (!Backtrack::validBoard(-1, -1)) {
        cout << "Initial board is invalid. Exiting now." << endl;

        return;
    }

    if (Backtrack::doneBoard()) {
        cout << "Board is already solved:" << endl;

        Backtrack::printBoard();

        return;
    } else {
        cout << "Solving board now." << endl;
    }


    if (Backtrack::solveHelper()) {
        cout << "Board is solvable:" << endl;

        Backtrack::printBoard();
    } else {
        cout << "Board is unsolvable." << endl;
    }
}

