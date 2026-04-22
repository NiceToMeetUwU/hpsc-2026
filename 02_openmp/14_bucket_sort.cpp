#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

int main() {
    int n = 500; 
    int range = 10;
    std::vector<int> key(n);
    
    // 1. initialize
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        key[i] = rand() % range;
        printf("%d ",key[i]);
    }
    printf("\n");
    std::vector<int> bucket(range, 0);

    // 2. frequency
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        #pragma omp atomic
        bucket[key[i]]++;
    }

    // 3. offset
    std::vector<int> offset(range, 0);
    for (int i = 1; i < range; i++) {
        offset[i] = offset[i - 1] + bucket[i - 1];
    }

    // 4.bucket time
    std::vector<int> bucket_count = bucket; 
    
    #pragma omp parallel for
    for (int i = 0; i < range; i++) {
        int start_pos = offset[i];
        int count = bucket_count[i];
        for (int j = 0; j < count; j++) {
            key[start_pos + j] = i;
        }
    }

    
    for (int i = 0; i < n; i++) printf("%d ", key[i]);
    printf(".sorted\n");

    return 0;
}