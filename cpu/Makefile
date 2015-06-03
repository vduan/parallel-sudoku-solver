CC = g++

CFLAGS = -c -Wall -g

all: backtrack

backtrack: backtrack.o checkSudoku.o
	$(CC) backtrack.o checkSudoku.o -o checkSudoku

checkSudoku.o: checkSudoku.cpp
	$(CC) $(CFLAGS) checkSudoku.cpp

backtrack.o: backtrack.cpp
	$(CC) $(CFLAGS) backtrack.cpp

clean:
	rm -rf *o *~ checkSudoku
