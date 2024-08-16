# Text-Summarisation
This repository contains the implementation of a text summarization project using various deep learning models, including LSTM-based encoder-decoder, T5, and BART. The project focuses on building a robust text summarizer and evaluating its performance using ROUGE-L scores. Additionally, it includes a mechanism to generate explanations of the summarization process using cross-attention scores from T5 and BART models.

**Project Overview**
Models Implemented
LSTM-based Encoder-Decoder: A traditional sequence-to-sequence model using Long Short-Term Memory (LSTM) networks for both encoding and decoding the input text.
T5 (Text-To-Text Transfer Transformer): A transformer-based model that converts all NLP tasks into a text-to-text format, making it versatile for various text generation tasks, including summarization.
BART (Bidirectional and Auto-Regressive Transformers): A transformer model that combines the strengths of BERT and GPT, designed for generating high-quality text summaries.
**Performance Metrics**
ROUGE-L Scores:
BART: Achieved a ROUGE-L score of 44.2.
T5: Achieved a ROUGE-L score of 50.
These scores serve as baseline benchmarks for the effectiveness of the models in generating accurate and concise text summaries.
**Explanation Generation**
Cross-Attention Scores: For T5 and BART, the project includes the generation of explanations for the summarization process by utilizing cross-attention scores. These scores help in understanding how different parts of the input text influence the generated summary, providing insights into the decision-making process of the models.
