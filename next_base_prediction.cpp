#include "utils.h"
#include <iostream>

// Usage: ./next_base_prediction sample.fasta [order]
// Trains Markov models by group and prints top predictions for a given context.

int main(int argc, char** argv) {
    if (argc < 2) { cerr << "Usage: " << argv[0] << " sample.fasta [order]\n"; return 1; }
    string fasta = argv[1];
    int order = (argc >= 3) ? stoi(argv[2]) : 3;
    auto entries = read_fasta(fasta);
    if (entries.empty()) { cerr << "No sequences.\n"; return 1; }

    // group like before
    map<string, vector<string>> groups;
    for (auto &p : entries) {
        string h = p.first; for (auto &c : h) c = toupper((unsigned char)c);
        string label = "unknown";
        if (h.find("HUMAN") != string::npos) label = "human";
        else if (h.find("CHIMP") != string::npos) label = "chimp";
        groups[label].push_back(p.second);
    }

    // train Markov per group
    map<string, Markov> mmaps;
    for (auto &g : groups) mmaps.emplace(g.first, Markov(order, 1.0));
    for (auto &g : groups) {
        auto &mm = mmaps[g.first];
        for (auto &s : g.second) mm.train_sequence(s);
    }

    string context;
    cout << "Enter context (last " << order << " bases) for prediction (e.g. GAT): ";
    if (!getline(cin, context) || context.size() == 0) {
        context = "GAT";
        cout << "Using default context: " << context << "\n";
    }
    for (auto &p : mmaps) {
        cout << "\nSpecies: " << p.first << " predictions:\n";
        auto probs = p.second.probs_for_context_str(context);
        vector<pair<char,double>> v;
        for (int i=0;i<4;++i) v.push_back({int_to_base(i), probs[i]});
        sort(v.begin(), v.end(), [](auto &a, auto &b){ return a.second > b.second; });
        for (auto &pr : v) cout << " " << pr.first << "(" << pr.second << ")";
        cout << "\n";
    }
    return 0;
}
