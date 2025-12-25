#include "utils.h"
#include <iostream>
#include <algorithm>



int main(int argc, char** argv) {
    if (argc < 3) { cerr << "Usage: " << argv[0] << " sample.fasta kmer [threshold_percentile]\n"; return 1; }
    string fasta = argv[1];
    string qk = argv[2];
    double thr = (argc >= 4) ? stod(argv[3]) : 1.0;
    int k = (int)qk.size();

    auto entries = read_fasta(fasta);
    if (entries.empty()) { cerr << "No sequences.\n"; return 1; }

    /
    KmerCounts ref(k);
    for (auto &p : entries) ref.add_sequence(p.second);

    
    vector<double> freqs;
    freqs.reserve(ref.counts.size());
    long double total = 0;
    for (auto c : ref.counts) total += c;
    for (auto c : ref.counts) freqs.push_back((total>0) ? (double)(c / total) : 0.0);

    
    int64_t code = ref.encode_kmer_str(qk);
    double qfreq = (code >= 0) ? ref.freq_of_kmer_code((uint64_t)code) : 0.0;

    
    uint64_t less = 0;
    for (auto f : freqs) if (f < qfreq) ++less;
    double perc = 100.0 * ((double)less / (double)freqs.size());
    bool rare = (perc <= thr);

    cout << "K-mer: " << qk << " freq=" << qfreq << " percentile=" << perc << "% threshold=" << thr << "%\n";
    cout << "Status: " << (rare ? "RARE" : "COMMON") << "\n";
    return 0;
}
