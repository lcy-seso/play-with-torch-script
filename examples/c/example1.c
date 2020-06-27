#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "utils.h"

void bi_direction_rnn(  // stacked bidirectional rnn.
    int batch_size, int depth, int max_seq_len,
    float input[batch_size][max_seq_len], int seq_lens[batch_size],
    float output1[batch_size][max_seq_len][depth] /* write buffer1 */,
    float output2[batch_size][max_seq_len][depth] /* write buffer2*/,
    float output[batch_size][max_seq_len][depth] /*write buffer3*/) {
  float x;
  float h_prev;
  float h;
  int seq_len;

#pragma scop
  for (int i = 0; i < batch_size; ++i) {  // parallel loop
    for (int j = 0; j < depth; ++j) {     // over sub-graphs
      seq_len = seq_lens[i];              // data-dependent loop-bounds
      for (int k = 0; k < seq_len; ++k) {
        if (j == 0) {
          x = input[i][k];
        } else {
          x = output1[i][k][j - 1];
        }

        if (k == 0) {
          h_prev = 0.;
        } else {
          h_prev = output1[i][k - 1][j];
        }

        // cells1[k](h_prev, x)
        output1[i][k][j] = cell(h_prev, x);
      }

      for (int k = seq_len - 1; k >= 0; --k) {
        if (j == 0) {
          x = input[i][seq_len - k];
        } else {
          x = output2[i][k][j - 1];
        }

        if (k == seq_len - 1) {
          h_prev = 0.;
        } else {
          h_prev = output2[i][j][k + 1];
        }
        // cells1[k](h_prev, x) is not supported.
        output2[i][k][j] = cell(h_prev, x);
      }

      for (int n = 0; n < seq_len; ++n)
        output[i][n][j] = output1[i][n][j] + output2[i][seq_len - 1 - n][j];
    }
  }
#pragma endscop
}

int main() {
  srand(time(NULL));

  int batch_size = 4;
  int depth = 3;

  int Min_Len = 4;
  int Max_Len = 20;

  int seq_lens[batch_size];
  for (int i = 0; i < batch_size; ++i)
    seq_lens[i] = rand() % (Max_Len - Min_Len) + Min_Len;

  int max_seq_len = max_element(seq_lens, batch_size);
  int min_seq_len = min_element(seq_lens, batch_size);

  // Gen random input data
  float input[batch_size][max_seq_len];
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < max_seq_len; ++j) {
      if (j < max_seq_len) {
        input[i][j] = (float)rand() / (float)(RAND_MAX);
      } else {
        input[i][j] = 0.;
      }
    }
  }

  // initialize output buffer.
  float output1[batch_size][max_seq_len][depth];
  float output2[batch_size][max_seq_len][depth];
  float output[batch_size][max_seq_len][depth];
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < max_seq_len; ++j) {
      for (int k = 0; k < depth; ++k) {
        output1[i][j][k] = 0.;
        output2[i][j][k] = 0.;
        output[i][j][k] = 0.;
      }
    }
  }

  bi_direction_rnn(batch_size, depth, max_seq_len, input, seq_lens, output1,
                   output2, output);

  for (int i = 0; i < batch_size; ++i) {
    int seq_len = seq_lens[i];
    for (int j = 0; j < seq_len; ++j) {
      for (int k = 0; k < depth; ++k) {
        printf("%.2f, ", output[i][j][k]);
      }
      printf("\n");
    }
    printf("\n");
  }
}
