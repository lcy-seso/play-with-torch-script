#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "utils.h"

void foo(int seq_len, const float input[seq_len],
         float output1[seq_len] /* write buffer1 */,
         float output2[seq_len] /* write buffer2*/,
         float output[seq_len] /*write buffer3*/) {
#pragma scop
  for (int i = 1; i <= seq_len; ++i) {  // forward
    output1[i] = cell(input[i - 1], output1[i - 1]);
  }

  for (int i = seq_len; i >= 1; --i) {  // backward
    output2[i] = cell(input[i - 1], output2[i + 1]);
  }

  for (int i = 1; i <= seq_len; ++i) {
    output[i] = output1[i] + output2[i];
  }
#pragma endscop
}

void foo1(int seq_len, float input[seq_len],
          float output1[seq_len] /* write buffer1 */,
          float output2[seq_len] /* write buffer2*/,
          float output[seq_len] /*write buffer3*/) {
#pragma scop
  for (int i = 1; i <= seq_len; ++i) {
    output1[i] = cell(input[i - 1], output1[i - 1]);
    output2[seq_len + 1 - i] =
        cell(input[seq_len - i], output2[seq_len - i + 2]);
  }
  for (int i = 1; i <= seq_len; ++i) {
    output[i] = output1[i] + output2[i];
  }
#pragma endscop
}

int main() {
  srand(time(NULL));

  int Min_Len = 4;
  int Max_Len = 20;
  int seq_len = rand() % (Max_Len - Min_Len) + Min_Len;

  // Gen random input data
  float input[seq_len];
  for (int i = 0; i < seq_len; ++i) {
    input[i] = (float)rand() / (float)(RAND_MAX);
  }

  // the first and the last element are for initial states.
  float output1[seq_len + 2];
  float output2[seq_len + 2];
  float output[seq_len + 2];

  for (int i = 0; i < seq_len + 2; ++i) {
    output1[i] = 0.;
    output2[i] = 0.;
    output[i] = 0.;
  }

  foo(seq_len, input, output1, output2, output);
  for (int i = 1; i <= seq_len; ++i) {
    printf("%.2f, ", output[i]);
  }
  printf("\n");

  for (int i = 0; i < seq_len + 2; ++i) {
    output1[i] = 0.;
    output2[i] = 0.;
    output[i] = 0.;
  }
  foo1(seq_len, input, output1, output2, output);
  for (int i = 1; i <= seq_len; ++i) {
    printf("%.2f, ", output[i]);
  }
  printf("\n");
}
