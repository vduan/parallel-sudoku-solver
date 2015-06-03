#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define _USE_MATH_DEFINES
// Source: 
// http://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


__global__
void
cudaDFSKernel(float *nodes, float *values, float *stacks, int target, int bound, float *max_val) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    while (index < bound) {
        stacks[index*100*3] = nodes[index];
        stacks[index*100*3+1] = values[index];
        stacks[index*100*3+2] = 15;
        
        int top = 1;
        float c_max = *max_val;

        while (top != 0) {

            // s.pop() equivalent
            top = top - 1;
            float n = stacks[index*100*3+top*3];   
            float v = stacks[index*100*3+top*3+1];   
            int l = stacks[index*100*3+top*3+2];

            if (l == target) {
                if (v > c_max) {
                    c_max = v;
                }
                continue;
            }
            else if (((l * v + (target - l)) / target) <= c_max) {
                continue;
            }
            else {
                float n1 = fmod(2.5 * n, 1);
                float n2 = n1 + 0.5;

                float v1 = (l * v + cos(2 * 3.14159265358979323 * n1)) / (l + 1);
                float v2 = (l * v + cos(2 * 3.14159265358979323 * n2)) / (l + 1);

                stacks[index*100*3+top*3] = n1;
                stacks[index*100*3+top*3+1] = v1;
                stacks[index*100*3+top*3+2] = l+1;

                top += 1;
                
                stacks[index*100*3+top*3] = n2;
                stacks[index*100*3+top*3+1] = v2;
                stacks[index*100*3+top*3+2] = l+1;

                top += 1;
            }
        }    
        
        atomicMax(max_val, c_max);
        
        index += blockDim.x * gridDim.x;
    }
    
}

void cudaCallDFSKernel(const unsigned int blocks,
                    const unsigned int threadsPerBlock,
                    float *nodes, 
                    float *values,
                    float *stacks,
                    const unsigned int target, 
                    const unsigned int bound, 
                    float *max_val) {
    cudaDFSKernel<<<blocks, threadsPerBlock>>> 
        (nodes, values, stacks, target, bound, max_val);
}

__global__
void
cudaBFSKernel(float *nodes, float *values, float *n_out, float *v_out,
                int level, int bound, float* max_val) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (index < bound) {
        float child_1 = fmod(2.5 * nodes[index], 1);
        float child_2 = fmod(child_1 + 0.5, 1);

        float v_1 = (level * values[index] + cos(2 * M_PI * child_1)) / (level + 1);
        float v_2 = (level * values[index] + cos(2 * M_PI * child_2)) / (level + 1);

        n_out[2*index] = child_1;
        n_out[2*index + 1] = child_2;
        v_out[2*index] = v_1;
        v_out[2*index + 1] = v_2;
        
        atomicMax(max_val, v_1);
        atomicMax(max_val, v_2);

        index += blockDim.x * gridDim.x;
    }

}

void cudaCallBFSKernel(const unsigned int blocks,
                        const unsigned int threadsPerBlock,
                        float *nodes,
                        float *values,
                        float *n_out,
                        float *v_out,
                        const unsigned int level,
                        const unsigned int bound,
                        float *max_val) {
    cudaBFSKernel<<<blocks, threadsPerBlock>>>
        (nodes, values, n_out, v_out, level, bound, max_val);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "wrong number of args\n\n");
        fprintf(stderr, "3 args should be: \n \
                        <l, i.e. level to search to> \n \
                        <threads per block> \n \
                        <number of blocks> \n");
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock = atoi(argv[2]);
    int blocks = atoi(argv[3]);

    // levels to iterate through
    int target = atoi(argv[1]);
    
    // bfs target
    int target_level = 15;

    // how many are left after we cut some branches
    //int total;

    // gpu storage
    float *values_dev;
    float *nodes_dev;
    float *out_dev;
    float *outv_dev;
    float *max_val;
    
    float *out = (float *) malloc(pow(2, target_level - 1)*sizeof(float));
    float *outv = (float *) malloc(pow(2, target_level -1)*sizeof(float));
    float *values = (float *) malloc(sizeof(float));
    float *nodes = (float *) malloc(sizeof(float));
    float *result = (float *) malloc(sizeof(float)); 
    // realloc() !!!!
    
    values[0] = cos(2 * M_PI * 0.5);
    nodes[0] = 0.5;
    
    // host storage
    cudaMalloc((void**) &values_dev, pow(2, target_level - 1)*sizeof(float));
    cudaMalloc((void**) &nodes_dev, pow(2, target_level - 1)*sizeof(float));

    cudaMalloc((void**) &out_dev, pow(2, target_level - 1)*sizeof(float));
    cudaMalloc((void**) &outv_dev, pow(2, target_level - 1)*sizeof(float));
    
    cudaMalloc((void**) &max_val, sizeof(float));

    cudaMemcpy(values_dev, values, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(nodes_dev, nodes, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(max_val, -2.0, sizeof(float));
    
    int bound;
    for (int level = 1; level < target_level; level++) {
        bound = pow(2, level - 1);
        cudaCallBFSKernel(blocks, threadsPerBlock, nodes_dev, values_dev, out_dev, outv_dev, level, bound, max_val);
        cudaMemcpy(nodes_dev, out_dev, pow(2, level)*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(values_dev, outv_dev, pow(2, level)*sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    // gpu storage for dfs
    float *stacks;
    
    cudaMalloc((void**) &stacks, pow(2, target_level - 1)*100*(sizeof(float)+sizeof(float)+sizeof(int)));

    bound = pow(2, target_level - 1);
    
    for (int i = 16; i <= 100; i++) {
        cudaMemset(max_val, -2.0, sizeof(float));
        cudaCallDFSKernel(blocks, threadsPerBlock, out_dev, outv_dev, stacks, i, bound, max_val);  

        cudaMemcpy(result, max_val, sizeof(float), cudaMemcpyDeviceToHost);
        printf("level %d gives a max_val of: %f\n", i, *result);
    }


    cudaFree(values_dev);
    cudaFree(nodes_dev);
    cudaFree(out_dev);
    cudaFree(outv_dev);
    cudaFree(max_val);
    cudaFree(stacks);

    free(values);
    free(nodes);
    free(out);
    free(outv);
    free(result);
    return 0;
}
