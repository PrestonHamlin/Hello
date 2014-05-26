# While a simple one-line statement would get the job done, I thought it would
#   be a good idea to do things in a manner more representative of how I see a
#   language is used or what it is good for. Python has a multitude of such
#   things. As such, I chose generators from a hat. Ok, that was a lie. I don't
#   have a hat. I picked them out of a fishbowl.

# Code by Preston Hamlin

def HelloGen():
    s   = "Hello World!"
    itr, length = 0, len(s)

    while itr < length:
        yield s[itr]
        itr += 1
    yield ""                            # just to mark terminal case


def main():
#    print('Hello World!')
    g = HelloGen()
    for c in g:
        print(c)

if __name__ == "__main__":
    main()