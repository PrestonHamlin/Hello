// Implementation file for HelloClass class as well as a main routine to demo
//  the class. Normally I have the driver file separate. However, for the sake
//  of brevity I integrated the code.

// Code by Preston Hamlin

#include "hello.h"

int main() {
  HelloClass hello;         // will go out of scope with no issues

  return 0;
}





HelloClass::HelloClass() {
  SayHello();
}
HelloClass::~HelloClass() {
  SayGoodbye();
}

void HelloClass::SayHello() {
  printf("\nHello World!");
}
void HelloClass::SayGoodbye() {
  printf("\nGoodbye Cruel World!\n");
}