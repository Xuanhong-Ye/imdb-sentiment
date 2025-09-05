# imdb-sentiment
# IMDB Sentiment Classification

This repository contains PyTorch implementations of several models for **IMDB movie review sentiment classification**, including **BERT**, **LSTM**, and **fully-connected (FC)** models. The project is designed for text classification tasks and demonstrates training, evaluation, and performance visualization.

---

## 📂 Project Structure
imdb_sentiment/
├── models/ # Pretrained models and saved trained models
│ └── bert-base-uncased/
├── result/ # Training metrics plots
├── data/ # Datasets (not included due to size)
├── imdb_bert_model.py # BERT model and dataset definitions
├── imdb_lstm_model.py # LSTM model definition
├── imdb_fc_model.py # Fully-connected model definition
├── train_bert_model.py # Training script for BERT
├── train_lstm_model.py # Training script for LSTM
├── train_fc_model.py # Training script for FC model
├── .gitignore
└── README.md



---

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- NumPy
- scikit-learn
- Matplotlib

Install the required packages via pip:

```bash
pip install torch transformers numpy scikit-learn matplotlib
