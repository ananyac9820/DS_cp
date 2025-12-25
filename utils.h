#ifndef UTILS_H
#define UTILS_H

#include <bits/stdc++.h>
using namespace std;


inline int base_to_int(char b) {
    switch (b) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;
    }
}
inline char int_to_base(int v) {
    static const char m[4] = {'A','C','G','T'};
    return m[v & 3];
}


static inline vector<pair<string,string>> read_fasta(const string &path) {
    vector<pair<string,string>> out;
    ifstream ifs(path);
    if (!ifs) {
        cerr << "Error: cannot open FASTA " << path << "\n";
        return out;
    }
    string line, header, seq;
    while (getline(ifs, line)) {
        if (line.size() == 0) continue;
        if (line[0] == '>') {
            if (!header.empty()) {
                out.push_back({header, seq});
                seq.clear();
            }
            header = line.substr(1);
        } else {
            for (char c : line) if (!isspace((unsigned char)c)) seq.push_back(c);
        }
    }
    if (!header.empty()) out.push_back({header, seq});
    return out;
}


struct KmerCounts {
    int k;
    size_t vocab_size;
    vector<uint64_t> counts; 

    KmerCounts(int k_ = 4) : k(k_) {
        if (k < 1) k = 1;
   
        vocab_size = (k <= 31) ? (1ULL << (2 * k)) : 0;
        if (vocab_size == 0) throw runtime_error("k too large for dense k-mer vector");
        counts.assign(vocab_size, 0ULL);
    }

    void add_sequence(const string &seq) {
        int n = (int)seq.size();
        if (n < k) return;
        uint64_t mask = ( (k*2) < 64 ) ? ((1ULL << (2*k)) - 1ULL) : ~0ULL;
        uint64_t code = 0;
        int valid = 0;
        for (int i = 0; i < n; ++i) {
            int v = base_to_int(seq[i]);
            if (v >= 0) {
                code = ((code << 2) | (uint64_t)v) & mask;
                ++valid;
            } else {
                code = 0;
                valid = 0;
            }
            if (valid >= k) counts[(size_t)code] += 1;
        }
    }

    vector<double> freq_vector() const {
        vector<double> vec(counts.size(), 0.0);
        long double total = 0.0L;
        for (auto c : counts) total += c;
        if (total <= 0.0L) return vec;
        for (size_t i = 0; i < counts.size(); ++i) vec[i] = (double)(counts[i] / total);
        return vec;
    }

    uint64_t total_counts() const {
        uint64_t s = 0;
        for (auto &c : counts) s += c;
        return s;
    }

 
    int64_t encode_kmer_str(const string &s) const {
        if ((int)s.size() != k) return -1;
        int64_t code = 0;
        for (int i = 0; i < k; ++i) {
            int v = base_to_int(s[i]);
            if (v < 0) return -1;
            code = (code << 2) | v;
        }
        return code;
    }

    
    double freq_of_kmer_code(uint64_t code) const {
        long double total = 0.0L;
        for (auto c : counts) total += c;
        if (total <= 0.0L) return 0.0;
        if (code >= counts.size()) return 0.0;
        return double(counts[code] / total);
    }
};


static inline double cosine_similarity(const vector<double> &a, const vector<double> &b) {
    if (a.size() != b.size()) return 0.0;
    long double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (long double)a[i] * (long double)b[i];
        na += (long double)a[i] * (long double)a[i];
        nb += (long double)b[i] * (long double)b[i];
    }
    if (na <= 0.0L || nb <= 0.0L) return 0.0;
    return (double)(dot / (sqrt(na) * sqrt(nb)));
}


struct Markov {
    int order;
    double laplace;

    unordered_map<uint64_t, array<uint64_t,4>> trans;
    unordered_map<uint64_t, uint64_t> context_total;

    Markov(int order_ = 3, double laplace_ = 1.0) : order(order_), laplace(laplace_) {
        if (order < 1) order = 1;
    }

    void train_sequence(const string &seq) {
        int n = (int)seq.size();
        if (n <= order) return;
        uint64_t mask = ( (order*2) < 64 ) ? ((1ULL << (2*order)) - 1ULL) : ~0ULL;
        uint64_t rolling = 0;
        int valid = 0;
        for (int i = 0; i < n; ++i) {
            int v = base_to_int(seq[i]);
            if (v < 0) { rolling = 0; valid = 0; continue; }
            rolling = ((rolling << 2) | (uint64_t)v) & mask;
            ++valid;
            if (valid > order) {
                uint64_t context = rolling >> 2;    // last 'order' bases (except current)
                int nextb = v;
                auto &arr = trans[context];
                arr[nextb] += 1;
                context_total[context] += 1;
            }
        }
    }

    array<double,4> probs_for_context(uint64_t context) const {
        array<double,4> out{};
        uint64_t tot = 0;
        auto it = context_total.find(context);
        if (it != context_total.end()) tot = it->second;
        auto it2 = trans.find(context);
        for (int b = 0; b < 4; ++b) {
            uint64_t c = 0;
            if (it2 != trans.end()) c = it2->second[b];
            out[b] = ( (double)c + laplace ) / ( (double)tot + laplace * 4.0 );
        }
        return out;
    }

    array<double,4> probs_for_context_str(const string &context_str) const {
        if ((int)context_str.size() < order) {
            array<double,4> uniform; for (int i=0;i<4;++i) uniform[i] = 1.0/4.0; return uniform;
        }
        uint64_t code = 0;
        for (int i = (int)context_str.size() - order; i < (int)context_str.size(); ++i) {
            int v = base_to_int(context_str[i]);
            if (v < 0) { array<double,4> uniform; for (int j=0;j<4;++j) uniform[j]=1.0/4.0; return uniform; }
            code = (code << 2) | (uint64_t)v;
        }
        return probs_for_context(code);
    }

    double prob_transition(const string &context_str, char next_base) const {
        int v = base_to_int(next_base);
        if (v < 0) return 0.0;
        auto p = probs_for_context_str(context_str);
        return p[v];
    }
};

#endif 
