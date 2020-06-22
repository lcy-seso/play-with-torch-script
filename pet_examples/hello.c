void func(int N, float A[]) {
#pragma scop
  for (int i = 1; i < N; ++i) {
    A[i - 1] *= 2;
    A[i] = foo(A[i - 1], 2.3);
  }
#pragma endscop
}
