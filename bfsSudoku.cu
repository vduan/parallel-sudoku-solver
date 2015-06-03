#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>



__global__
void
cudaBFSKernel(int *old_boards, int *new_boards, int total_boards, int *board_index, int *empty_spaces, int *empty_space_count) {
    
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // board_index must start at zero 

    while (index < total_boards) {
        //printf("index is %d\n", index);
        // find the next empty spot
        int found = 0;
        for (int i = (index * 81); (i < (index * 81) + 81) && (found == 0); i++) {
            //printf("checking index number %d, number there is %d\n", i, old_boards[i]);
            // found a open spot
            if (old_boards[i] == 0) {
                found = 1;
                // get the correct row and column shits
                //printf("open spot found is %d\n", i);
                int temp = i - 81 * index;
                int row = temp / 9;
                int col = temp % 9;
                
                
                //printf("row and column of open spot is %d, %d\n", row, col);
                // figure out which numbers work here
                for (int attempt = 1; attempt <= 9; attempt++) {
                    //printf("trying the number: %d\n", attempt);
                    int works = 1;
                    // row constraint, test various columns
                    for (int c = 0; c < 9; c++) {
                        if (old_boards[row * 9 + c + 81 * index] == attempt) {
                            //printf("attempt number %d breaks here M\n", attempt);
                            works = 0;
                        }
                    }
                    // column contraint, test various rows
                    for (int r = 0; r < 9; r++) {
                        if (old_boards[r * 9 + col + 81 * index] == attempt) {
                            //printf("attempt number %d breaks here J\n", attempt);
                            works = 0;
                        }
                    }
                    // box constraint
                    for (int r = 3 * (row / 3); r < 3; r++) {
                        for (int c = 3 * (col / 3); c < 3; c++) {
                            if (old_boards[r * 9 + c + 81 * index] == attempt) {
                            //printf("attempt number %d breaks here T for row, col = (%d,%d)\n", attempt, r, c);
                                works = 0;
                            }
                        }
                    }
                    if (works == 1) {
                        //printf("the number: %d works\n", attempt);
                        // copy the whole board

                        int next_board_index = atomicAdd(board_index, 1);
                        int empty_index = 0;
                        for (int r = 0; r < 9; r++) {
                            for (int c = 0; c < 9; c++) {
                                new_boards[next_board_index * 81 + r * 9 + c] = old_boards[index * 81 + r * 9 + c];
                                if (old_boards[index * 81 + r * 9 + c] == 0 && (r != row || c != col)) {
                                    empty_spaces[empty_index + 81 * next_board_index] = r * 9 + c;

                                    empty_index++;
                                    printf("row and column of open spot is %d, %d\n", r, c);
                                    printf("empty spots: %d\n", empty_index);
                                }
                            }
                        }
                        empty_space_count[next_board_index] = empty_index;
                        printf("the skleeg is : %d\n", empty_index);
                        new_boards[next_board_index * 81 + row * 9 + col] = attempt;
                    }

                }
            }
        }
        index += blockDim.x * gridDim.x;
    }

}

void callBFSKernel(const unsigned int blocks, 
                        const unsigned int threadsPerBlock,
                        int *old_boards,
                        int *new_boards,
                        int total_boards,
                        int *board_index,
                        int *empty_spaces,
                        int *empty_space_count) {
    cudaBFSKernel<<<blocks, threadsPerBlock>>>
        (old_boards, new_boards, total_boards, board_index, empty_spaces, empty_space_count);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("asdf");
        exit(EXIT_FAILURE);
    }

    const unsigned int threadsPerBlock = atoi(argv[1]);
    const unsigned int blocks = atoi(argv[2]);
    const unsigned int iterations = atoi(argv[3]);
    // initial starting
    // allocate worst case board sizes

    int *new_boards;
    int *old_boards;
    int *empty_spaces;
    int *empty_space_count;

    int n = 40000000; // = 5^9
    cudaMalloc(&empty_spaces, n * sizeof(int));
    cudaMalloc(&empty_space_count, (n / 81 + 1) * sizeof(int));
    cudaMalloc(&new_boards, n * sizeof(int));
    cudaMalloc(&old_boards, n * sizeof(int));
    // where to store the next new board generated
    int *board_index;
    // same as board index, except we need to set board_index to zero every time and this can stay
    int total_boards = 1;
    cudaMalloc(&board_index, sizeof(int));
    cudaMemset(board_index, 0, sizeof(int));
    cudaMemset(new_boards, 0, n * sizeof(int)); 
    int *sudokus = (int*) malloc(n * sizeof(int));
    memset(sudokus, 0, n * sizeof(int));
    // the actual bound
    //memset(sudokus, 0, n* sizeof(int));
int first[81] = {0,0,0,0,3,7,6,0,0,
                 0,0,0,6,0,0,0,9,0,
                 0,0,8,0,0,0,0,0,4,
                 0,9,0,0,0,0,0,0,1,
                 6,0,0,0,0,0,0,0,9,
                 3,0,0,0,0,0,0,4,0,
                 7,0,0,0,0,0,8,0,0,
                 0,1,0,0,0,9,0,0,0,
                 0,0,2,5,4,0,0,0,0};
            
    cudaMemcpy(old_boards, first, 81 * sizeof(int), cudaMemcpyHostToDevice);
    callBFSKernel(blocks, threadsPerBlock, old_boards, new_boards, total_boards, board_index, empty_spaces, empty_space_count);
    
    int *host_count = (int*) malloc(sizeof(int));

   
    for (int i = 0; i < iterations; i++) {
        cudaMemcpy(host_count, board_index, sizeof(int), cudaMemcpyDeviceToHost);
        printf("total boards after an iteration %d: %d\n", i, *host_count);
        cudaMemcpy(old_boards, new_boards, 81 * (*host_count) * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemset(board_index, 0, sizeof(int));
        cudaMemset(new_boards, 0, n * sizeof(int)); 

        callBFSKernel(blocks, threadsPerBlock, old_boards, new_boards, (*host_count), board_index, empty_spaces, empty_space_count);
    }
    cudaMemcpy(host_count, board_index, sizeof(int), cudaMemcpyDeviceToHost);
    printf("new number of boards retrieved is %d\n", *host_count);
    cudaMemcpy(sudokus, new_boards, 81 * sizeof(int), cudaMemcpyDeviceToHost);   

    
//    for (int i = 0; i < 81; i++) {
//        if (i % 81 == 0) {
//            printf("\n\nNEW BOARD!\n");
//        }
//        if (i % 9 == 0 && (i % 81 != 0)) {
//            printf("\n");
//        }
//        printf("%d ", sudokus[i]);
//        
//    }
    cudaMemcpy(host_count, empty_space_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("empty = %d\n", *host_count);

    cudaFree(board_index);
    cudaFree(new_boards);
    cudaFree(old_boards);
    free(sudokus);

}

