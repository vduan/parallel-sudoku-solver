CC = g++

CFLAGS = -c -Wall -g

all: sudoku

sudoku: checkSudoku.o
	$(CC) checkSudoku.o -o checkSudoku

checkSudoku.o: checkSudoku.cpp
	$(CC) $(CFLAGS) checkSudoku.cpp

clean:
	rm -rf *o *~ checkSudoku
