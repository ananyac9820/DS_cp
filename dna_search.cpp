#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cstring> 

using namespace std;

struct Suffix {
    int index;
    const char* text; 
};

bool compareSuffixes(const Suffix& a, const Suffix& b) {
    return strcmp(a.text, b.text) < 0;
}

void search(string pattern, const string& dna, const vector<int>& suffixArray) {
    int m = pattern.length();
    int n = dna.length();
    int left = 0, right = n - 1;
    vector<int> results;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        int res = dna.compare(suffixArray[mid], m, pattern);

        if (res == 0) {
            results.push_back(suffixArray[mid]);
            int temp = mid - 1;
            while(temp >= 0 && dna.compare(suffixArray[temp], m, pattern) == 0) {
                results.push_back(suffixArray[temp]);
                temp--;
            }
            temp = mid + 1;
            while(temp < n && dna.compare(suffixArray[temp], m, pattern) == 0) {
                results.push_back(suffixArray[temp]);
                temp++;
            }
            break; 
        }

        if (res < 0) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    if (results.empty()) {
        cout << "\nPattern '" << pattern << "' not found." << endl;
    } else {
        cout << "\nPattern '" << pattern << "' found at " << results.size() << " locations. Examples:" << endl;
        sort(results.begin(), results.end());
        for (int i=0; i<min((int)results.size(), 10); ++i) { // Print up to 10 results
            cout << results[i] << " ";
        }
        cout << endl;
    }
}

int main() {
    string fileName = "sample.fasta";
    ifstream inputFile(fileName);
    string fullDna = "";
    string line;

    if (!inputFile.is_open()) {
        cout << "Error: Could not open " << fileName << endl;
        return 1;
    }

    while (getline(inputFile, line)) {
        if (line.empty() || line[0] == '>') {
            continue;
        }
        fullDna += line;
    }
    inputFile.close();

    cout << "Successfully read " << fullDna.length() << " DNA bases." << endl;

    int n = fullDna.length();
    vector<Suffix> suffixes(n);

    for (int i = 0; i < n; i++) {
        suffixes[i] = {i, fullDna.c_str() + i};
    }

    sort(suffixes.begin(), suffixes.end(), compareSuffixes);

    vector<int> suffixArray;
    suffixArray.reserve(n);
    for (int i = 0; i < n; i++) {
        suffixArray.push_back(suffixes[i].index);
    }
    
    cout << "Suffix Array built successfully." << endl;

    search("TATTA", fullDna, suffixArray);
    search("GGGG", fullDna, suffixArray);
    search("GATTACA", fullDna, suffixArray);

    return 0;
}