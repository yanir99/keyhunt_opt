#ifndef AVX2_SECP256K1_H
#define AVX2_SECP256K1_H

#include <immintrin.h>
#include <stdint.h>

// Field element structure (256-bit)
typedef struct {
    uint64_t d[4];
} secp256k1_fe;

// Jacobian point structure
typedef struct {
    secp256k1_fe x, y, z;
} secp256k1_gej;

// Function declarations
void secp256k1_fe_add_avx2(secp256k1_fe* r, const secp256k1_fe* a, const secp256k1_fe* b);
void secp256k1_fe_mul_avx2(secp256k1_fe* r, const secp256k1_fe* a, const secp256k1_fe* b);
void secp256k1_fe_sqr_avx2(secp256k1_fe* r, const secp256k1_fe* a);
void secp256k1_gej_double_avx2(secp256k1_gej* r, const secp256k1_gej* a);
void secp256k1_ecmult_batch_avx2(secp256k1_gej* results, const uint64_t* scalars, size_t count);

#endif
