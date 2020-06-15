#include <stdio.h>

int A[5];
typedef int A5[5];

void func(int len, A5 B[]) {
#pragma scop
  for (int i = 0; i < len; ++i) {
    for (int j = 0; j < 5; ++j) {
      B[i][j] = i + j;
    }
  }
#pragma endscop
}

int main() {
  int len = 7;
  A5 B[len];

  func(len, B);

  for (int i = 0; i < len; ++i) {
    for (int j = 0; j < 5; ++j) {
      printf("%d ", B[i][j]);
    }
    printf("\n");
  }
  return 0;
}
