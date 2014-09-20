# This file contains several FizzBuzz methods. Not much else to say.
#   So... having a good day? That's cool. Maybe you could check out some of the
#   other code I wrote. The stuff here is not all that exciting. To me, I mean.
#   If you have a thing for FizzBuzz and find it fascinating then by all means
#   contact me and we can negotiate the licensing of my code, so you can print
#   it out, make paper aeroplanes or otherwise decorate your office or other 
#   establishment. Maybe make a swan with a different function on each wing.

# Code by Preston Hamlin

import sys

# Conditional implementation, the obvious enumeration of possibilities.
#   Granted, this could be simplified by not having to worry about the newlines
#   print() will insert, but that is beside the point. Although, I suppose this
#   implementation may be better than some naive ones since it does not compute
#   modulo 15.
def FizzBuzzCond(n):
    for i in range(1, n+1):
        f = i%3
        b = i%5
        if ((f == 0) and (b == 0)):
            print("FizzBuzz")
        elif (f == 0):
            print("Fizz")
        elif (b == 0):
            print("Buzz")
        else:
            print(i)
            

# Array implementation.
def FizzBuzzArray(n):
    k = n+1                         # so that "n+1" is not in all these loops
    stuff = ["" for i in range(k)]  # str(i) would be overwritten often (46%)
    for i in range(0, k, 3):
        stuff[i] += "Fizz"
    for i in range(0, k, 5):
        stuff[i] += "Buzz"
    for i in range(1, k):
        if stuff[i]:
            print(stuff[i])
        else:
            print(str(i))           # array 0-indexed
            
        
        
# Functional implementation, no variables with the exception of the iteration.
#   I prefer this method. For some reason I never see this from other people or
#   on the web. Maybe I should go back to LISP or learn Haskell.
def Fizz(val):
    if val%3 == 0:
        sys.stdout.write("Fizz")
        return 1
    return 0

def Buzz(val):
    if val%5 == 0:
        sys.stdout.write("Buzz")
        return 1
    return 0

def FizzBuzzFunc(n):
    for i in range(1, n+1):
        if (Fizz(i) | Buzz(i)):     # binary OR to prevent short-circuit eval 
            sys.stdout.write("\n")
        else:
            sys.stdout.write(str(i) + "\n")




def main():
    print("\n\n=== Conditional ===")
    FizzBuzzCond(50)
    print("\n\n=== Array ===")
    FizzBuzzArray(50)
    print("\n\n=== Functional ===")
    FizzBuzzFunc(50)
    
if __name__ == "__main__":
    main()
    
