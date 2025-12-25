#include "utils.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " sample.fasta [k]\n";
        return 1;
    }
    string fasta = argv[1];
    int k = (argc >= 3) ? stoi(argv[2]) : 4;

    auto entries = read_fasta(fasta);
    if (entries.empty()) { cerr << "No sequences.\n"; return 1; }

    map<string, vector<string>> groups;
    for (auto &p : entries) {
        string hdr = p.first;
        string s = p.second;
        string h = hdr;
        for (auto &c : h) c = toupper((unsigned char)c);
        string label = "unknown";
        if (h.find("HUMAN") != string::npos) label = "human";
        else if (h.find("CHIMP") != string::npos || h.find("CHIMPANZEE") != string::npos) label = "chimp";
        groups[label].push_back(s);
    }

    cout << "Groups found:\n";
    for (auto &g : groups) cout << " - " << g.first << " : " << g.second.size() << "\n";


    map<string, KmerCounts> kmaps;
    for (auto &g : groups) kmaps.emplace(g.first, KmerCounts(k));
    for (auto &g : groups) {
        auto &kc = kmaps[g.first];
        for (auto &seq : g.second) kc.add_sequence(seq);
    }


    map<string, vector<double>> freq;
    for (auto &p : kmaps) freq[p.first] = p.second.freq_vector();

    string query = entries[0].second;
    if (query.size() > 200) query = query.substr(0,200);
    KmerCounts qkc(k);
    qkc.add_sequence(query);
    auto qvec = qkc.freq_vector();

    cout << "\nClassification result for first sequence snippet:\n";
    string best = "";
    double best_score = -1e9;
    for (auto &p : freq) {
        double s = cosine_similarity(qvec, p.second);
        cout << " - " << p.first << " : cosine = " << s << "\n";
        if (s > best_score) { best_score = s; best = p.first; }
    }
    cout << "Predicted species: " << best << " (score=" << best_score << ")\n";
    return 0;
}
