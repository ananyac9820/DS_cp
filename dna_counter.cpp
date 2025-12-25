#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <queue>

using namespace std;


struct Node {
    char data;
    int count;
    Node *left;
    Node *right;

    Node(char data, int count) {
        this->data = data;
        this->count = count;
        left = right = nullptr;
    }
};

struct compare {
    bool operator()(Node* l, Node* r) {
        return (l->count > r->count);
    }
};

void storeCodes(Node* root, string str, map<char, string> &codes) {
    if (root == nullptr) return;
    if (root->data != '$') codes[root->data] = str;
    storeCodes(root->left, str + "0", codes);
    storeCodes(root->right, str + "1", codes);
}

void encodeFile(string inputFileName, string outputFileName, map<char, string> &codes) {
    ifstream inputFile(inputFileName);
    ofstream outputFile(outputFileName);
    string line;
    cout << "Encoding file..." << endl;
    while (getline(inputFile, line)) {
        if (line.empty() || line[0] == '>') continue;
        for (char base : line) {
            char upper_base = toupper(base);
             if (codes.find(upper_base) != codes.end()) {
                outputFile << codes[upper_base];
            }
        }
    }
    inputFile.close();
    outputFile.close();
    cout << "Encoding complete." << endl;
}

void decodeFile(string inputFileName, string outputFileName, Node* root) {
    ifstream inputFile(inputFileName);
    ofstream outputFile(outputFileName);
    Node* curr = root;
    string s;
    inputFile >> s;
    cout << "Decoding file..." << endl;
    for (int i = 0; i < s.size(); i++) {
        if (s[i] == '0') curr = curr->left;
        else curr = curr->right;
        if (curr->left == nullptr && curr->right == nullptr) {
            outputFile << curr->data;
            curr = root;
        }
    }
    inputFile.close();
    outputFile.close();
    cout << "Decoding complete." << endl;
}

int main() {
    map<char, int> counts;
    string inFileName = "sample.fasta";
    ifstream inputFile(inFileName);
    string line;
    while (getline(inputFile, line)) {
        if (line.empty() || line[0] == '>') continue;
        for (char base : line) {
            char upper_base = toupper(base);
            if (upper_base == 'A' || upper_base == 'C' || upper_base == 'G' || upper_base == 'T') {
                counts[upper_base]++;
            }
        }
    }
    inputFile.close();
    
    priority_queue<Node*, vector<Node*>, compare> pq;
    for (auto pair : counts) pq.push(new Node(pair.first, pair.second));
    
    while (pq.size() != 1) {
        Node* left = pq.top(); pq.pop();
        Node* right = pq.top(); pq.pop();
        Node* top = new Node('$', left->count + right->count);
        top->left = left;
        top->right = right;
        pq.push(top);
    }
    
    Node* root = pq.top();
    map<char, string> codes;
    storeCodes(root, "", codes);
    
    encodeFile(inFileName, "encoded.txt", codes);
    decodeFile("encoded.txt", "decoded.fasta", root);
    
    return 0;
}

