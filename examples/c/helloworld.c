void foo(int N, float A[N]) {
#pragma scop
  A[0] = 1.5;
  for (int i = 1; i < N; ++i) {  // forward
    A[i] = A[i - 1] + 2.;
  }
#pragma endscop
}
