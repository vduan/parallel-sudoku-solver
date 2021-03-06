# Parallelized Sudoku Solver on the GPU
###### Victor Duan and Michael Teng
###### CS 179: GPU Programming Caltech Spring 2015

### Summary

We implemented a parallelized CUDA program that can efficiently solve a Sudoku puzzle using a
backtracking algorithm.

### Background

##### Introduction to Sudoku

Sudoku is a popular puzzle game usually played on a 9x9 board of numbers between 1 and 9.

![alt text](https://github.com/vduan/cs179sudoku/blob/master/res/img/ex_sudoku_board.png?raw=true "Example Sudoku Board")

The goal of the game is to fill the board with numbers. However, each row can only contain one of
each of the numbers between 1 and 9. Similarly, each column and 3x3 sub-board can only contain one
of each of the numbers between 1 and 9. This makes for an engaging and challenging puzzle game.

A standard Sudoku puzzle may have about 50-60 empty spaces to solve for. A brute force algorithm
would have an incredibly large search space to wade through. In fact, the task of solving a Sudoku
puzzle is NP-complete.

##### Solving Algorithm

A common algorithm to solve Sudoku boards is called backtracking. This algorithm is essentially a
depth first search in the tree of all possible guesses in the empty space of the Sudoku board. The
algorithm finds the first open space, and tries the number 1 there. If that board is valid, it will
continue trying values in the other open spaces. If the board is ever invalid, backtrack by undoing
the most recent guess and try another value there. If all values have been tried, backtrack again.
If the board is valid and there are no more empty spaces, the board has been solved! If the
backtracking goes back to the first empty space and there are no more possible values, we have tried
every possible combination of numbers on the board and there is no solution. We can illustrate this
more clearly with the pseudocode of the algorithm here:

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

##### Parallelization Challenges

At first glance, this algorithm, and indeed depth first search in general, does not appear to be
very parallelizable. This is because the algorithm depends on a stack. In the recursive case, the
stack is implicit in the function call stack. Stacks are hard to parallelize because threads cannot
all work off of the same stack and productively move forward in the algorithm together without
causing very high contention for access to the stack.

##### Alternative Solutions
###### Simulated Annealing

Simulated annealing is a heuristic to solve optimization problems using probabilistic methods. The
basic idea is to begin by filling the Sudoku board with the numbers. These numbers need not be in
the correct place, but there should be exactly nine of each number from 1-9 on the board.
Additionally, each of the 3x3 sub-boards should have every number from 1-9. The algorithm then
proceeds as follows:

1. Randomly choose one of the sub-boxes and swap two of the non-fixed numbers.
2. If the board is "better", leave it as is and continue.
3. If the board is not "better", probabilistically change it back to the original board.

Now, what do we mean by better? We can score the boards so that the algorithm has an optimization in
mind. One potential scoring system is to score a board by adding -1 for each unique number in each
row and column. Thus, if a row has 9 unique values, it contributes -9 to the score. The board is
then solved if the score is -162, which corresponds to 9 unique values in every row and column and
each sub-box has 9 unique values from the start.

This algorithm typically needs to be run for thousands of generations to find a solution. However,
we can see that many of these swaps may not really interact with each other (if they are in
different sub-boards). Additionally, there is not really an order that needs to be preserved. Thus,
this looks like a parallelizable approach to solving Sudoku puzzles that is ideal for the GPU.

###### Dancing Links

The Dancing Links algorithm was devloped by Donald Knuth to solve the exact cover problem. Without
getting into the details, the exact cover problem takes a matrix of 0's and 1s and searches for a
set of rows such that in those rows, each column contains exactly one non-zero value. The problem is
then to express Sudoku as an exact cover problem. We can see that Sudoku can be described as a set
of constraints for each location of the board:

1. Each space may only contain a single value.
2. Each row may only contain each number one time.
3. Each column may only contain each number one time.
4. Each sub-board may only contain each number one time.

We can express these constraints in a single row of length N * N * 4 where N = 9 for a standard
Sudoku board and 4 is for the 4 constraints. We must also have a row for potential space and number,
which is N * N * N rows. This gives us a matrix of size N^3 x 4N^2. The initial state of the board
can be expressed by selecting which rows are in the exact cover. The output of the Dancing Links
algorithm will give us a subset of rows where each constraint is filled exactly once. Each row
represents a space-number pairing, so the exact cover will be a solution to the Sudoku puzzle.

However, the Dancing Links algorithm does not appear to be a very parallelizable solution either. At
its core, it is similar to the depth first search solution and will face similar parallelization
challenges.

###### Parallelization over different branches of the depth first search

Finally, we can modify the depth first search approach so that threads can function independently
without all reading from the same stack. Breadth first search is easy to parallelize, so we can
begin by finding all possible valid boards that fill the first, say, 20 empty spaces of the given
puzzle. This may give us something like thousands of possible boards. These boards may then be
passed to another kernel, where each thread then applies the backtracking algorithm to it's own
board. If any thread finds a solution, all threads stop, and that solution is passed back to the
cpu.

We can see that both of these steps are much easier to parallelize than the original depth first
search solution. In the end, this is the primary approach we chose to focus on.

### Approach: Parallel Backtracking

##### Algorithm

The parallel approach to backtracking can be broken down into two main steps:

1. Breadth first search from the beginning board to find all possible boards with the first X empty
spaces filled. This will return Y possible starting boards for the next step of the algorithm.

2. Attempt to solve each of these Y boards separately in different threads on the GPU. If a solution
is found, terminate the program and return the solution.

The speedup of this approach is in the parallelization of part 2. This allows us to search through
the depth first search approach in parallel, which allows for great speedup.

We can do each of these steps in separate kernels.

##### Board Generation Kernel

This kernel will expect the following things as input:

- the previous frontier of boards from which we will be searching
- a pointer to where the next layer of boards will be stored
- total number of boards in the previous frontier
- a pointer to where the indices of the empty spaces in the next layer of boards will be stored
- a pointer to where the number of empty spaces in each board is stored

The kernel will spawn threads where each one generates the children of the board. This can be done 
without knowing the actual graph, because the children board of each input is simply the possible 
boards given the first empty space. For example, if there is an empty and the numbers, 1,2,3,6,9 
all work, this thread will add 5 new boards to the new boards array. For difficult sudokus, worst case
the board starts to grow by a factor of 9 per level. Of course, in practice, this is never the case, 
but given this, we have set a heuristic on how far the BFS will go. For us, we use the 20th level, as
this will result in about ~50 thousand, which is a nice number to send to DFS, given that there is
64k registers on the average single gpu (Kepler). 

##### Backtracking Kernel

This kernel will expect the following things as input:

- all the boards to search through
- number of boards to search through
- location of the empty spaces
- number of empty spaces

The kernel will then spawn threads that each individually handle one board at a time. It will do the
classic backtracking algorithm as described earlier.

It is important to note that recursion has some issues on GPU programming. While it is supported for
compute capability >=2.0, we found that implementing it resulted in strange behavior, even for a
single thread. The same code may be run on the cpu and run just fine. Thus, we switched from using
the implicit stack of recursive depth first search (function call stack) to an explicit stack of
depth first search, where we try values at each empty space and backtrack. This is why we also
include the location of the empty spaces and the number of empty spaces total.

We also use a global finished flag so that when a solution is found, all threads are notified and
the kernel can terminate.

This kernel allows us to work on these boards at (# of threads) the speed because that is how many
boards we can process at once.


### Results

In general, we find that using our parallelized backtracking algorithm results in speed ups on  the
order of 10. For our two sample boards, we found that the cpu backtracing, which is entirely
sequential results in ~4 milliseconds and ~3 seconds for the easy and hard boards, respectively.  On
the gpu, we find that kernel calls take ~1 millisecond and ~800 milliseconds for the easy and hard
boards, respectively. We also ran analyses on the backtracking algorithm for a single board to the
kernel in order to get a better benchmark for how much BFS speedup is. The kernels for a  single
board, and thus single thread DFS is about 30 times slower, than when we parallelize on the 20th
level of the search space.

### References

1. For insights on applying Knuth's Dancing Links algorithm to solving Sudoku puzzles:
https://www.ocf.berkeley.edu/~jchu/publicportal/sudoku/sudoku.paper.html

2. For insights on the simulated annealing approach to solving Sudoku puzzles: "Sudoku Using
Parallel Simulated Annealing" by Zahra Karimi-Dehkordi, Kamran Zamanifar, Ahmad Baraani-Dastjerdi,
Nasser Ghasem-Aghaee. http://link.springer.com/chapter/10.1007/978-3-642-13498-2_60#




### How to run parallel Sudoku solver

1. To compile the code, navigate to the src/ directory and run make. The executable will appear in
the bin/ directory.

2. To run the solver, run the executable with arguments for threads per block, max number of blocks,
and filename of the puzzle. We have supplied some sample boards in the res/sample_inputs/ directory.
You can, of course, create your own boards to test. The numbers in the file should be 0 for empty,
and between 1 and 9 for filled spaces.

```
cd SUDOKU_DIR/src/
make
SUDOKU_DIR/bin/CudaSudoku 512 256 SUDOKU_DIR/res/sample_inputs/hard_1.txt
```