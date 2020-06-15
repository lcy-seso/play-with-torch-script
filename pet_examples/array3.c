#include <stdio.h>

int A[5];
typedef int A5[5];

typedef struct SA {
  A5 b;
  float f;
} SA;

void func() {
#pragma scop
  int len = 7;
  SA B[len];

  for (int i = 0; i < len; ++i) {
    for (int j = 0; j < 5; ++j) {
      B[i].b[j] = i + j;
    }
  }
#pragma endscop

  for (int i = 0; i < len; ++i) {
    for (int j = 0; j < 5; ++j) {
      printf("%d ", B[i].b[j]);
    }
    printf("\n");
  }
}

int main() {
  func();
  return 0;
}
