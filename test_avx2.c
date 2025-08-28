#include "avx2_secp256k1.h"
#include <stdio.h>
#include <time.h>

void test_avx2_performance() {
    const int NUM_TESTS = 1000000;
    secp256k1_fe a, b, result;
    
    // Initialize test data
    for (int i = 0; i < 4; i++) {
        a.d[i] = 0x123456789ABCDEF0ULL + i;
        b.d[i] = 0xFEDCBA9876543210ULL + i;
    }
    
    clock_t start = clock();
    
    for (int i = 0; i < NUM_TESTS; i++) {
        secp256k1_fe_mul_avx2(&result, &a, &b);
    }
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("AVX2 Performance Test:\n");
    printf("Multiplications: %d\n", NUM_TESTS);
    printf("Time: %.2f seconds\n", time_taken);
    printf("Rate: %.0f ops/second\n", NUM_TESTS / time_taken);
}

int main() {
    test_avx2_performance();
    return 0;
}