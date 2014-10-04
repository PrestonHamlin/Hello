/* wordcount - mutex-free word counting
Code by Preston Hamlin

This program launches a series of threads to read segments of a text file for
  the purpose of counting the number of occurances of each word. Each thread
  is provided its own copy of working variables, so there is no need for an
  item such as a mutex.
This is simply a demonstration of the usage of STL threads.

TODO: time reads
TODO: wrap into WordCounter class?
*/

#include <iostream>
#include <fstream>
#include <cctype>
#include <map>
#include <string>
#include <thread>

using namespace std;

void FatalError(string msg);
void StripString(string &str);
void PartitionedRead(int id, map<string, int> *wordcount, int *locations);
void SkipToWord(ifstream &ifile, int &loc);

const int NUM_THREADS = 100;


int main() {
  // sequential read variables
  ifstream          ifile;
  string            word;
  map<string, int>  wordcount;

  // concurrent read variables, one per thread
  thread            threads[NUM_THREADS];
  int               locations[NUM_THREADS+1];
  int               file_length;
  map<string, int>  wordcounts[NUM_THREADS];
  map<string, int>  wordcount_total;


  // open file for sequential read and calculation of partition locations
  ifile.open("../raven.txt");
  if (!ifile) FatalError("cannot open file for reading");
  ifile.seekg(0, ios::end);
  file_length = ifile.tellg();
  locations[0] = 0;

  for (int i = 1; i <= NUM_THREADS; ++i) {
    // set rough location
    locations[i] = (file_length*i)/NUM_THREADS; // integer division, I know  
    if (i == NUM_THREADS) break;
    SkipToWord(ifile, locations[i]);

    // single word spans an entire partition, partitions too small
    if (locations[i] == locations[i - 1]) FatalError("too many threads");
    else locations[i] = ifile.tellg();
  }
//  for (int i = 1; i <= NUM_THREADS; ++i) cout << '\n' << locations[i];
  ifile.seekg(0); // re-seek to beginning


  // perform sequential read
  while (ifile >> word) {
    StripString(word);

    if (word.empty()) continue;
    else wordcount[word]++;
  }

  // sequential read results
  // for (auto& pair : wordcount) {
  //   cout << '\n' << pair.first << ": " << pair.second;
  // }



  // perform concurrent read
  for (int i = 0; i < NUM_THREADS; ++i) { // launch threads
    threads[i] = thread(PartitionedRead, i, &(wordcounts[i]), locations);
  }
  cout << "\nthreads created";
  for (int i = 0; i < NUM_THREADS; ++i) { // terminate thread
    threads[i].join();
  }
  for (int i = 0; i < NUM_THREADS; ++i) { // merge results
    for (auto& pair : wordcounts[i]) {
      wordcount_total[pair.first] += pair.second;
    }
  }
  cout << "\nthreads joined";

  // concurrent read results
  for (auto& pair : wordcount_total) {
    cout << '\n' << pair.first << ": " << pair.second;
  }



  // cleanup
  ifile.close();

  cout << "\n\nHave a nice day\n";
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
    if ((*itr >= 'A') && (*itr <= 'Z')) { // if upper case letter
      *itr += 32;                         // convert to lower case
      ++itr;
    }
    else if (((*itr >= 'a') && (*itr <= 'z')) ||
             (*itr == '\'')
            )
    {
      ++itr;
    }
    else {
      str.erase(itr);
    }
  }
}


// function run by multiple threads
void PartitionedRead(int id, map<string, int> *wordcount, int *locations) {
  string word;
  ifstream ifile;

  // light debugging output, not exactly mutex worthy
//  cout << id << '\n';

  ifile.open("../raven.txt");
  if (!ifile) FatalError("cannot open file for readings");
  
  ifile.seekg(locations[id]); 
  
  // read until section ends or EOF
  while ((ifile >> word) && (ifile.tellg() <= locations[id+1])) {
    StripString(word);

    if (word.empty()) continue;
    else (*wordcount)[word]++;        // ->at() throws a mean old exception
  }

  // light debugging output, not exactly mutex worthy
//  cout << "\nthread " << id << " finished";

  ifile.close();
}


// helper function that moves read pointer up to beginning of new word
void SkipToWord(ifstream &ifile, int &loc) {
  char c;

  // check if location is good (at start of word)
  ifile.seekg(loc-1);
  c = ifile.peek();
  if (iswspace(c)) return;  // loc is good

  // seek until whitespace found, except for false last bound
  while (c = ifile.peek()) {
    cout << c;
    if (iswspace(c)) break;
    else if (c == 0) FatalError("too many threads");  // spans multiple threads
    else ifile.get();
  }
  // now skip whitespace
  while (c = ifile.peek()) {
    if (!iswspace(c)) break;
    else ifile.get();
  }

  // ifile read pointer now at beginning of word
  loc = ifile.tellg();
}

