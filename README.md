🧬 Hybrid Machine Learning and Algorithmic System for DNA Sequence Analysis

A hybrid bioinformatics system that combines high-performance C++ algorithms with Python-based machine learning models to perform DNA sequence analysis, next-base prediction, and species classification through a responsive Flask web interface.

📌 Project Overview

High-throughput genomic sequencing generates massive DNA data, making efficient analysis challenging. Traditional bioinformatics tools written in C++ are fast but rule-based, while machine learning models written in Python are powerful but often difficult to integrate and deploy.

This project bridges that gap by building a hybrid system that:

Uses C++ for fast, deterministic DNA analysis

Uses Machine Learning (LSTM & Neural Networks) for pattern recognition

Integrates everything through a Flask-based web application

The result is a system that is accurate, fast, and easy to use, even for non-programmers.

✨ Key Features

🔬 DNA sequence preprocessing and analysis

🔮 Next-base prediction using LSTM networks

🧪 Species classification using neural networks

⚡ High-performance C++ algorithmic core

🌐 Web-based interface for uploading and analyzing FASTA files

⏱️ End-to-end response time under 1 second

🏗️ System Architecture

The system follows a three-tier hybrid architecture:

User (Browser)
   ↓
Flask Web Interface (Python)
   ↓
--------------------------------
|  C++ Algorithmic Core         |
|  - Nucleotide counting        |
|  - Pattern analysis           |
|  - Statistical processing     |
--------------------------------
   ↓
Machine Learning Inference (Python / PyTorch)
   - LSTM Next-Base Predictor
   - Species Classifier


This modular design ensures scalability, maintainability, and performance.

🧠 Algorithms & Techniques Used
Classical / Algorithmic (C++)

Efficient nucleotide counting

Substring search

Grammar-based DNA structure analysis

k-mer rarity analysis

Statistical pattern extraction

Machine Learning (Python / PyTorch)

LSTM (Long Short-Term Memory) for next-base prediction

Feedforward Neural Network for species classification

One-hot encoding and sequence padding

Cross-Entropy loss with Adam optimizer

🧪 Dataset & Preprocessing

Data Source: NCBI RefSeq genomic database

Format: FASTA

Next-base prediction:

Sliding window of 50 bases → predict the 51st base

Species classification:

Full sequences converted to numerical vectors

Padding used for uniform input size

🚀 How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/ananyac9820/DS_cp.git
cd DS_cp

2️⃣ (Optional) Create Python environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

3️⃣ Run the Flask web app
python app.py

4️⃣ Open in browser
http://127.0.0.1:5000/


Upload a FASTA file and view predictions directly in the browser.

📊 Experimental Results
Metric	Performance
Next-base prediction accuracy	75–85%
Species classification accuracy	~90%
Average web request latency	< 1 second

These results demonstrate a strong balance between predictive accuracy and real-time usability.

📈 Key Observations

LSTM significantly outperforms random or simple frequency-based models

Lightweight ML models enable real-time inference

Hybrid design avoids the latency of heavy Transformer models

Web interface abstracts away algorithmic complexity for end users

🔮 Future Enhancements

Replace LSTM with Transformer-based models (e.g., DNABERT)

Add k-mer rarity and motif detection modules

Containerize system using Docker

Deploy on cloud platforms (AWS / Heroku)

Direct C++ inference using LibTorch

👥 Authors

Ananya Choudhari

Arya Bharat Patil

Aryan Bhat

Ankush Kumar

Department of Computer Engineering
Vishwakarma Institute of Technology, Pune
