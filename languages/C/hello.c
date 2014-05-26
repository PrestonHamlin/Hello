// This file contains some Hello World routines. Originally I was just going to
//  have the simple main routine, but I decided to alter some implementations
//  to better reflect some of the core aspects or methodologies which typify a
//  given language. In the case of C, that would be pointers.

// Code by Preston Hamlin

#include <stdio.h>

/*  simple main
int main() {
    printf("Hello World!");
    return 0;
}
*/

int main() {
  const char* const str   = "Hello World!";
  const char* itr         = str;

  do {
    printf("%s\n", itr);
  } while(*(++itr));

  return 0;
}