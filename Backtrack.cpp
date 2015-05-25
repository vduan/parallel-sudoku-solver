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


pair<int, int> Backtrack::findEmptySpot(int location) {
    for (int i = location + 1; i < N * N; i++) {
        if (board[i] == 0) {
            return make_pair<int, int>(i / N, i % N);
        }
    }

    return make_pair<int, int>(-1, -1);
}


bool Backtrack::solveHelper(int i, int j, int location) {
    cout << "solveHelper: i = " << i << ", j = " << j << ", location = " << location << endl;
    // i < 0 if this is the first call

    if (!validBoard(i, j)) {
        return false;
    }

    if (Backtrack::doneBoard()) {
        return true;
    }

    while (location < N * N) {
        // find a empty spot
        pair<int, int> next = Backtrack::findEmptySpot(location);
        int x = next.first;
        int y = next.second;
        location = x * N + y;

        if (x < 0) {
            return false;
        }

        for (int k = 1; k <= N; k++) {
            board[location] = k;
            if (Backtrack::solveHelper(x, y, location)) {
                return true;
            }
        }

        board[x * N + y] = 0;

        location++;
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
        cout << "Initial board is invalid." << endl;
    } else {
        cout << "Initial board is valid." << endl;
    }

    if (Backtrack::doneBoard()) {
        cout << "Board is already solved:" << endl;

        Backtrack::printBoard();

        return;
    } else {
        cout << "Solving board now." << endl;
    }


    if (Backtrack::solveHelper(-1, -1, 0)) {
        cout << "Board is solvable:" << endl;

        Backtrack::printBoard();
    } else {
        cout << "Board is unsolvable." << endl;
    }
}

