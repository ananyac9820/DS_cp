#include "utils.h"
#include <iostream>

// Usage: ./grammar_improbability sample.fasta prefix next_base [order] [threshold]
// Example: ./grammar_improbability sample.fasta GATT C 3 1e-4

int main(int argc, char** argv) {
    if (argc < 4) { cerr << "Usage: " << argv[0] << " sample.fasta prefix next_base [order] [threshold]\n"; return 1; }
    string fasta = argv[1];
    string prefix = argv[2];
    char nb = argv[3][0];
    int order = (argc >= 5) ? stoi(argv[4]) : 3;
    double thr = (argc >= 6) ? stod(argv[5]) : 1e-4;

    auto entries = read_fasta(fasta);
    if (entries.empty()) { cerr << "No sequences.\n"; return 1; }

    // train Markov on all sequences
    Markov mm(order, 1.0);
    for (auto &p : entries) mm.train_sequence(p.second);

    double p = mm.prob_transition(prefix, nb);
    bool flagged = p < thr;
    cout << "Transition: \"" << prefix << "\" -> '" << nb << "'  Probability=" << p << " threshold=" << thr << "\n";
    cout << "Flagged as improbable? " << (flagged ? "YES" : "NO") << "\n";
    return 0;
}
