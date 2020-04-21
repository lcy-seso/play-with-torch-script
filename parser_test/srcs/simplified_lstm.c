void StackedLSTM(int N, int M, int K) {
  int c[N][M][K];

#pragma scop
  for (int i = 0; i < N; ++i) {
    for (int j = 1; j < M; ++j) {
      for (int k = 1; k < K; ++k) {
        c[i][j][k] = c[i][j][k - 1] + c[i][j - 1][k];
      }
    }
  }
#pragma endscop
}
