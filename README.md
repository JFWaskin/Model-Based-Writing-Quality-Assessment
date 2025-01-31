# Model-Based Writing Quality Assessment

This Python script evaluates the **writing quality of multiple text files** using **language models**. It performs **perplexity analysis, sentiment evaluation, and sentence similarity measurement** to provide an overall assessment of textual coherence and fluency. The results are saved in a structured **CSV report**.
This program helps analyze the **quality of writing** by using artificial intelligence models. It reads text files, evaluates how fluent and well-structured the writing is, and saves the results in an easy-to-read report. The analysis focuses on three main aspects:

1. **Fluency (Perplexity Score)** – The program uses **GPT-2**, an AI model trained on vast amounts of text, to check how **predictable and natural** the writing feels. If a sentence flows smoothly and follows common patterns in the language, the AI assigns a lower score, meaning **better fluency**. Higher scores indicate **more awkward or complex** writing.

2. **Emotional Tone (Sentiment Score)** – The program applies a **sentiment analysis model** to determine whether the text feels **positive, neutral, or negative**. It assigns a score that averages the emotional tone of the whole text.

3. **Consistency (Sentence Similarity Score)** – Using **sentence embedding technology**, the program compares the similarity between sentences. This helps measure whether the writing **stays on topic and maintains a consistent style**, or if the ideas seem scattered.

### How It Works
1. The program scans a **folder of text files** and reads each file.
2. It runs each text through AI models to calculate **fluency, sentiment, and consistency**.
3. The results are saved in a **CSV file** (like an Excel spreadsheet), so you can easily review the scores.

### Why Use This?
- Helps **students, writers, and researchers** understand the strengths and weaknesses of their writing.
- Gives quick, **data-driven insights** into writing quality.
- Can be used to compare different writing styles or track **improvements over time**.

This tool is like an **AI-powered writing coach**, providing a scientific way to evaluate how well-written a piece of text is!
## Features
- **Perplexity Calculation:** Uses **GPT-2** to measure text fluency (lower perplexity = more fluent text).
- **Sentiment Analysis:** Uses **BERT-based sentiment classification** to gauge the emotional tone of the text.
- **Sentence Similarity Measurement:** Uses **Sentence Transformers** to compute the **semantic consistency** between sentences.
- **Batch Processing:** Analyzes all `.txt` files in a specified folder automatically.
- **CSV Output:** Saves the results in a structured format for easy analysis.

## Dependencies
- Python 3.8+
- `torch`
- `transformers`
- `sentence-transformers`
- `csv`
- `glob`
- `os`

## Installation
```bash
pip install torch transformers sentence-transformers
```

## Usage
1. **Prepare input files:** Place `.txt` files in the target directory.
2. **Run the script**:
   ```bash
   python writing_quality_assessment.py
   ```
3. **Output:** A CSV file (`model_based_writing_quality.csv`) with the following columns:
   - **File Name:** Name of the processed file
   - **Perplexity:** Measure of text fluency (lower is better)
   - **Sentiment Score:** Average sentiment score (higher means more positive sentiment)
   - **Avg Sentence Similarity:** Measures coherence and consistency

## Key Functions
- `tokenizer, model`: Loads **GPT-2** for perplexity computation.
- `sentiment_tokenizer, sentiment_model`: Uses **BERT** to analyze sentiment.
- `embedder.encode(sentences)`: Generates sentence embeddings to calculate similarity.
- `torch.exp(loss).item()`: Computes **perplexity** to assess fluency.
- `util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)`: Measures **semantic consistency**.

## Example Output
| File Name    | Perplexity | Sentiment Score | Avg Sentence Similarity |
|-------------|------------|----------------|-------------------------|
| text1.txt   | 30.25      | 0.67           | 0.82                    |
| text2.txt   | 45.12      | 0.45           | 0.78                    |

This tool is ideal for **automated writing evaluation, computational linguistics research, and NLP-driven text assessment.** 🚀
