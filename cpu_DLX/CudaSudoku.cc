#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>


#include "Sudoku.h"







int main(int argc, char* argv[]) {
    
    
  if (argc < 5){
      printf("Usage: (threads per block) (max number of blocks) (filename) (N) \n");
      exit(-1);
  }

  
  /* Additional parameters for the assignment */
  const unsigned int threadsPerBlock = atoi(argv[1]);
  const unsigned int maxBlocks = atoi(argv[2]);
  char *filename = argv[3];
  const int N = atoi(argv[4]);

  
  // const unsigned int blocks = std::min(maxBlocks, (unsigned int) ceil(
  //             numberOfNodes/float(threadsPerBlock)));


  printf("making solver\n");
  SudokuSolver solver(filename, N);

  printf("about to solve\n");
  solver.solve();
  
  return 0;
}
