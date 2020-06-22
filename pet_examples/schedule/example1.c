void foo(float A[5][7]) {
#pragma scop
  for (int i = 0; i < 5; ++i) {
    A[i][0] = 0.5;
    for (int j = 1; j < 7; ++j) {
      A[i][j] = A[i][j - 1] + j;
    }
  }
#pragma endscop
}
