// Simple FizzBuzz routine. For more elaborate comments and other goodies, go
//  check out the Python version.

// Code by Preston Hamlin

#include <stdio.h>

int Fizz(int val) {
  if (val%3 == 0) {
    printf("Fizz");
    return 1;
  }
  return 0;
}

int Buzz(int val) {
  if (val%5 == 0) {
    printf("Buzz");
    return 1;
  }
  return 0;
}

void FizzBuzz(int n) {
  for (int i=1; i<=n; ++i) {      // extract "int i" for older standards
    if (Fizz(i) | Buzz(i)) printf("\n");
    else printf("%i\n", i);
  }
}

int main() {
  FizzBuzz(50);
  return 0;     
}