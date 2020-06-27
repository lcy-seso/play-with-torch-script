
float cell(float h, float x) { return h + x; }

int max_element(const int* array, int size) {
  int max = array[0];
  for (int i = 1; i < size; ++i) max = array[i] > max ? array[i] : max;
  return max;
}

int min_element(const int* array, int size) {
  int min = array[0];
  for (int i = 1; i < size; ++i) min = array[i] < min ? array[i] : min;
  return min;
}
