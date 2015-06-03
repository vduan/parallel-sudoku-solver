CC = g++

CFLAGS = -c -Wall -g

all: Sudoku

Sudoku: Sudoku.o CudaSudoku.o
	$(CC) Sudoku.o CudaSudoku.o -o cudaSudoku

CudaSudoku.o: CudaSudoku.cc
	$(CC) $(CFLAGS) CudaSudoku.cc

Sudoku.o: Sudoku.cc
	$(CC) $(CFLAGS) Sudoku.cc

clean:
	rm -rf *o *~ cudaSudoku
