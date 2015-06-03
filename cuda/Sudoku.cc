#include "Sudoku.h"


void SudokuSolver::createMatrix() {
  int x, y;
  for (int a = 0; a < num_Col; a++) {
    for (int b = 0; b < num_Row; b++) {
      if (Data[a][b] != 0) {
        // set up the left, right, up, and down pointers in the toroidal linked list
        x = a; y = b;
        while (Data[x][y] == 0) {
          x = getLeft(x);
        }
        Matrix[a][b].left = &Matrix[x][y];

        x = a;
        while (Data[x][y] == 0) {
          x = getRight(x);
        }
        Matrix[a][b].right = &Matrix[x][y];

        x = a;
        while (Data[x][y] == 0) {
          y = getUp(y);
        }
        Matrix[a][b].up = &Matrix[x][y];

        y = b;
        while (Data[x][y] == 0) {
          y = getDown(y);
        }
        Matrix[a][b].down = &Matrix[x][y];

        // set the row header
        // this gets rewritten, but that is fine because we want the lowest one to be used
        RowHeader[b] = &Matrix[a][b];

        // set header pointer
        Matrix[a][b].Header = &Matrix[a][num_Row - 1];
        Matrix[a][b].IDNum = b;
      }
    }
  }

  for (int a = 0; a < num_Col; a++) {
    Matrix[a][num_Row - 1].IDNum = a;
  }

  // create root node
  Root->left = &Matrix[num_Col - 1][num_Row - 1];
  Root->right = &Matrix[0][num_Row - 1];
  Matrix[num_Col - 1][num_Row - 1].right = Root;
}


void SudokuSolver::cover(node *column) {
  node *row, *right;

  // remove column node from the linked list
  column->right->left = column->left;
  column->left->right = column->right;

  for (row = column->down; row != column; row = row->down) {
    for (right = row->right; right != row; right = right->right) {
      right->up->down = right->down;
      right->down->up = right->up;
    }
  }
}

void SudokuSolver::uncover(node *column) {
  node *row, *left;

  for (row = column->up; row != column; row = row->up) {
    for (left = row->left; left != row; left = left->left) {
      left->up->down = left;
      left->down->up = left;
    }
  }

  column->right->left = column;
  column->left->right = column;
}


void SudokuSolver::solutionRow(node *row) {
  SudokuSolver::cover(row->Header);

  for (node *right = row->right; right != row; right = right->right) {
    cover(right->Header);
  }
}


void SudokuSolver::printSolution() {
  int Sudoku[N][N];

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      Sudoku[i][j] = -1;
    }
  }

  for (int i = 0; i < nResult; i++) {
    Sudoku[getRow(Result[i])][getColumn(Result[i])] = getNumber(Result[i]);
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (i > 0 && i % 3 == 0 && j == 0) {
        for (int k = 0; k < N; k++) {
          printf("--");
        }
        printf("\n");
      }

      if (Sudoku[i][j] >= 0) {
        printf("%d%c", Sudoku[i][i] + 1, j % 3 == 2 ? '|' : ' ');
      } else {
        printf(". ");
      }
    }
    printf("\n");
  }
}


void SudokuSolver::search(int k) {
  if (Root->left == Root && Root->right == Root) {
    // done
    printf("------------------------\nsolved:\n");
    SudokuSolver::printSolution();
    done = true;
    return;
  }


  node *column = SudokuSolver::getFirstColumn();

  SudokuSolver::cover(column);

  for (node *row = column->down; row != column && !done; row = row->down) {
    // cover the row
    Result[nResult++] = row->IDNum;

    for (node *right = row->right; right != row; right = right->right) {
      SudokuSolver::cover(right->Header);
    }

    SudokuSolver::search(k + 1);

    // if it doesn't work
    for (node *right = row->right; right != row; right = right->right) {
      SudokuSolver::uncover(right->Header);
    }

    Result[--nResult] = 0;
  }

  SudokuSolver::uncover(column);
}


void SudokuSolver::createData() {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        int val = makeNumber(k, i, j);

        // set up the sparse matrix using the constraints for Sudoku
        Data[0 * N * N + getRc(val)][val] = 1;
        Data[1 * N * N + getNr(val)][val] = 1;
        Data[2 * N * N + getNc(val)][val] = 1;
        Data[3 * N * N + getNb(val)][val] = 1;
      }
    }
  }

  for (int i = 0; i < num_Col; i++) {
    Data[i][num_Row - 1] = 2;
  }

  SudokuSolver::createMatrix();
}


void SudokuSolver::loadPuzzle() {
  FILE *a_file = fopen(filename, "r");
  if (a_file == NULL) {
    printf("File load fail. Using empty board.\n");
    return;
  }

  char temp;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (!fscanf(a_file, "%c\n", &temp)) {
        printf("File loading error. Skipping rest of file.\n");
        return;
      }

      if (temp >= '1' && temp <= '9') {
        SudokuSolver::addNumber((temp - '1'), i, j);
      }
    }
  }
}


SudokuSolver::SudokuSolver(char *filename = 0, int N = 9)
    : filename(filename)
    , N(N)
    , nResult(0)
    , maxK(0)
{
  num_Col = 4 * N * N;
  num_Row = N * N * N + 1;  // 1 extra for header

  // const int nc = num_Col;
  // const int nr = num_Row;
  const int nc = 4 * 9 * 9;
  const int nr = 9 * 9 * 9 + 1;

  node **temp = new node*[nc];
  for (int i = 0; i < nc; i++) {
    Matrix[i] = new node[nr];
  }

  Matrix = temp;

  Root = new node();

  RowHeader = new node*[nr];

  Data = new char*[nc];
  for (int i = 0; i < nc; i++) {
    Matrix[i] = new node[nr];
  }

  Result = new int[nr];

}


SudokuSolver::~SudokuSolver() {
  for (int i = 0; i < num_Col; i++) {
    delete[] Matrix[i];
  }
  delete[] Matrix;
  delete Root;
  delete[] RowHeader;
  for (int i = 0; i < num_Col; i++) {
    delete[] Matrix[i];
  }
  delete[] Data;
  delete[] Result;
}


void SudokuSolver::solve() {
  SudokuSolver::createData();
  SudokuSolver::loadPuzzle();
  SudokuSolver::search(0);
}