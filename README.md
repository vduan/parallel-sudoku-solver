# Parallelized Sudoku Solver on the GPU
###### Victor Duan and Michael Teng
###### CS 179: GPU Programming Caltech Spring 2015

### Summary
We implemented a parallelized CUDA program that can efficiently solve a Sudoku puzzle using a backtracking algorithm.

### Background
Sudoku is a popular puzzle game usually played on a 9x9 board of numbers between 1 and 9. 

![alt text](https://raw.githubusercontent.com/vduan/cs179sudoku/master/report/ex_sudoku_board.png "Example of Sudoku Board")

The goal of the game is to fill the board with numbers. However, each row can only contain one of each of the numbers between 1 and 9. Similarly, each column and 3x3 sub-board can only contain one of each of the numbers between 1 and 9. This makes for an engaging and challenging puzzle game.

A common algorithm to solve Sudoku boards is called backtracking. This algorithm is essentially a depth first search in the tree of all possible guesses in the empty space of the Sudoku board. The algorithm finds the first open space, and tries the number 1 there. If that board is valid, it will continue trying values in the other open spaces. If the board is ever invalid, backtrack by undoing the most recent guess and try another value there. If all values have been tried, backtrack again. If the board is valid and there are no more empty spaces, the board has been solved! If the backtracking goes back to the first empty space and there are no more possible values, we have tried every possible combination of numbers on the board and there is no solution. We can illustrate this more clearly with the pseudocode of the algorithm here:

```
recursive_backtrack():
    if board is valid:
        index = index of first empty spot in board
        for value = 1 to 9:
            set board[index] = value
            if recursive_backtrack():
                return true;  // solved!
            set board[index] = 0
    // if we tried all values, or the board is invalid, backtrack
    return false;
```

At first glance, this algorithm, and indeed depth first search in general, does not appear to be very parallelizable. 