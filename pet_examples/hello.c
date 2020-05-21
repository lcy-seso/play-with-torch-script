void func(int N, float A[]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    A[i] *= 5.;
  }
#pragma endscop
}
