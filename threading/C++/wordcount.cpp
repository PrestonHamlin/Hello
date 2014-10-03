#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <thread>

using namespace std;

void FatalError(string msg);
void StripString(string &str);


int main() {
  // sequential read variables
  ifstream ifile;
  string word;
  map<string, int> wordcount;
  
  
  // perform sequential read
  ifile.open("../raven.txt");
  if (!ifile) FatalError("cannot open file for reading");
  
  while(ifile >> word) {
    StripString(word);
    
    if (word.empty()) continue;
    else wordcount[word]++;
  }
  
  // sequential read results
  map<string, int>::iterator itr = wordcount.begin();
  for (; itr != wordcount.end(); ++itr) {
    cout << itr->first << ": " << itr->second << '\n';
  }
  
  
  
  
  
  
  cout << "\nHave a nice day\n";
  return 0;
}



// simple error helper
void FatalError(string msg) {
  cout << "\nERROR: " << msg;
  exit(1);
}


// strips non-alphabetical characters from a string
// because regexes are overkill for such a simple stripping
void StripString(string &str) {
  string::iterator itr = str.begin();
  
  while (itr != str.end()) {
    if ((*itr >= 'A') && (*itr <= 'Z' )) {  // if upper case letter
      *itr += 32;                           // convert to lower case
      ++itr;
    }
    else if ( ((*itr >= 'a') && (*itr <= 'z' )) ||
              (*itr == '\'')
            ) {
      ++itr;
    }
    else {
      str.erase(itr);
    }
  }
}