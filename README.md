

### **Aim**
To implement a ticketing system where multiple threads simulate ticket allocation concurrently using CUDA.

---

### **Algorithm**
1. **Initialization:**
   - Define the total number of tickets (shared among threads).
   - Initialize an array to store ticket allocation for each thread.
2. **Thread Work:**
   - Each thread attempts to allocate tickets until all are sold.
   - Use atomic operations to ensure safe updates to shared variables.
3. **Kernel Execution:**
   - Launch CUDA kernel with multiple threads (representing ticket counters).
   - Use atomic operations for thread-safe ticket allocation.
4. **Result Retrieval:**
   - Copy the results (who allocated which ticket) from device to host memory.
   - Print the ticket allocation summary.

---

### **Program**

```cuda
#include <stdio.h>
#include <cuda.h>

#define TOTAL_TICKETS 100
#define NUM_THREADS 32
#define BLOCKS 4

__global__ void ticketingSystem(int *ticketsRemaining, int *allocation, int totalThreads) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x; // Unique thread ID
    while (atomicSub(ticketsRemaining, 1) > 0) { // Atomically decrement tickets
        atomicAdd(&allocation[threadId], 1); // Count tickets for this thread
    }
}

int main() {
    int ticketsRemaining = TOTAL_TICKETS;
    int totalThreads = BLOCKS * NUM_THREADS;

    // Allocate host memory
    int *h_allocation = (int *)malloc(totalThreads * sizeof(int));
    for (int i = 0; i < totalThreads; i++) {
        h_allocation[i] = 0;
    }

    // Allocate device memory
    int *d_ticketsRemaining, *d_allocation;
    cudaMalloc((void **)&d_ticketsRemaining, sizeof(int));
    cudaMalloc((void **)&d_allocation, totalThreads * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_ticketsRemaining, &ticketsRemaining, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_allocation, h_allocation, totalThreads * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    ticketingSystem<<<BLOCKS, NUM_THREADS>>>(d_ticketsRemaining, d_allocation, totalThreads);

    // Copy results back to host
    cudaMemcpy(h_allocation, d_allocation, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);

    // Display results
    printf("Ticket Allocation Summary:\n");
    for (int i = 0; i < totalThreads; i++) {
        printf("Thread %d allocated %d tickets\n", i, h_allocation[i]);
    }

    // Free memory
    cudaFree(d_ticketsRemaining);
    cudaFree(d_allocation);
    free(h_allocation);

    return 0;
}
```

---

### **Output**
- Displays the number of tickets allocated by each thread.

**Sample Output:**
```
Ticket Allocation Summary:
Thread 0 allocated 7 tickets
Thread 1 allocated 8 tickets
Thread 2 allocated 6 tickets
...
```

---

### **Result**
The program demonstrates a ticketing system where GPU threads represent ticket counters. Atomic operations ensure thread-safe ticket allocation. You can adjust the `TOTAL_TICKETS`, `NUM_THREADS`, and `BLOCKS` to observe scalability and performance.
