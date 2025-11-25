#include "test.h"

// original not aligned code 
// void test1(float *a, float *b, float *c, int N) {
//   __builtin_assume(N == 1024);

//   for (int i=0; i<I; i++) {
//     for (int j=0; j<N; j++) {
//       c[j] = a[j] + b[j];
//     }
//   }
// }



void test1(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c, int N) { // aligned code , 32 bytes 
  __builtin_assume(N == 1024);         
  a = (float *)__builtin_assume_aligned(a, 32);
  b = (float *)__builtin_assume_aligned(b, 32);
  c = (float *)__builtin_assume_aligned(c, 32);

  for (int i = 0; i < I; ++i) {
    for (int j = 0; j < N; ++j) {
      c[j] = a[j] + b[j];
    }
  }
}
