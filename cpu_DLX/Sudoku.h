#ifndef SUDOKU_H
#define SUDOKU_H


#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>



class SudokuSolver {

private:
  struct node {

    int IDNum;

    node *Header;

    node *left;
    node *right;
    node *up;
    node *down;
  };

  char *filename; // filename to load puzzle from
  int N;          // dimensions of board (e.g. 9x9 for standard board)
  int num_Col;    // number of columns in the sparse matrix for exact cover (4 N^2)
  int num_Row;    // number of rows in the sparse matrix for exact cover (N^3)

  node **Matrix;
  node *Root;
  node **RowHeader;   // header for each row
  char **Data;  // data from loaded board
  int *Result;
  int nResult;
  bool done;
  int maxK;

  // shortcuts to accessing data in the algorithm
  inline int getLeft(int i)  { return i - 1 < 0 ? num_Col - 1 : i - 1; }
  inline int getRight(int i) { return (i + 1) % num_Col; }
  inline int getUp(int i)    { return i - 1 < 0 ? num_Row - 1 : i - 1; }
  inline int getDown(int i)  { return (i + 1) % num_Row; }

  // gets the next column to cover
  inline node *getFirstColumn() { return Root->right; }

  // a point on the board may be described by its number, row, and column. suppose it was
  // formatted to one number like so: number * N^2 + row * N + column
  // these functions extract the pieces from such a number as well as the subboard it belongs to
  // inline int getNumber(int x) { return x  / (N * N); }
  // inline int getRow(int x) { return (x / N) % N; }
  // inline int getColumn(int x) { return x % 9; }
  // inline int getSubboard(int x) { return ((getRow(x) / 3) * 3) + (getColumn(x) / 3); }

  // inline int getRc(int x) { return getRow(x) * 9 + getColumn(x); }
  // inline int getNr(int x) { return getNumber(x) * 9 + getRow(x); }
  // inline int getNc(int x) { return getNumber(x) * 9 + getColumn(x); }
  // inline int getNb(int x) { return getNumber(x) * 9 + getSubboard(x); }

  inline int retNb(int N) { return N/81; }
  inline int retRw(int N) { return (N/9)%9; }
  inline int retCl(int N) { return N%9; }
  inline int retBx(int N) { return ((retRw(N)/3)*3) + (retCl(N)/3); }
  inline int retSq(int N) { return retRw(N)*9 + retCl(N); }
  inline int retRn(int N) { return retNb(N)*9 + retRw(N); }
  inline int retCn(int N) { return retNb(N)*9 + retCl(N); }
  inline int retBn(int N) { return retNb(N)*9 + retBx(N); }

  // given number, row, and column, create a compressed number
  inline int makeNumber(int number, int row, int column) { return number * N * N + row * N + column; }

  inline void addNumber(int number, int row, int column) {
    int val = makeNumber(number, row, column);
    solutionRow(RowHeader[val]);
    maxK++;
    Result[nResult++] = val;
  }

  // build toroidal linked list for Dancing Links algorithm
  void createMatrix();

  // removes the column and its rows from the toroidal linked list
  void cover(node *column);

  // puts the column and its rows back into the toroidal linked list
  void uncover(node *column);

  void solutionRow(node *row);

  // print the solution
  void printSolution();

  // searh algorithm to find a solution to the sudoku board
  void search(int k);

  // build the sparse matrix for Dancing Links algorithm
  void createData();

  // load a puzzle from a file
  void loadPuzzle();

public:
  // constructors and destructors
  SudokuSolver(char *filename, int N);
  ~SudokuSolver();

  // main driver to solve a sudoku board
  void solve();

};


#endif  // SUDOKU_H
