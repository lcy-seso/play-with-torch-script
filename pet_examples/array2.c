#include <stdio.h>

void func() {
  int len = 7;
  int A[5];
  typedef int A5[5];

#pragma scop
  A5 B[len];

  for (int i = 0; i < len; ++i) {
    for (int j = 0; j < 5; ++j) {
      B[i][j] = i + j;
    }
  }
#pragma endscop

  for (int i = 0; i < len; ++i) {
    for (int j = 0; j < 5; ++j) {
      printf("%d ", B[i][j]);
    }
    printf("\n");
  }
}

int main() {
  func();
  return 0;
}
