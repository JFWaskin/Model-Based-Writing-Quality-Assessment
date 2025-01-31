import glob
import os
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

directory_path = "/Users/jonathanwaskin/Desktop/Study Methodologies/untitled folder/data/在线构思"
file_paths = glob.glob(os.path.join(directory_path, "*.txt"))

output_csv_path = "/Users/jonathanwaskin/Desktop/model_based_writing_quality.csv"

if os.path.exists(output_csv_path):
    os.remove(output_csv_path)

# 困惑度模型
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

# 加载情感分析模型
sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment").to(device)

# 加载句子嵌入模型
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps" if torch.backends.mps.is_available() else "cpu")

# 保存检测结果的列表
results = []

# 分析每篇文章的写作质量
for file_path in file_paths:
    with open(file_path, "r") as file:
        text = file.read()

    # 1. 计算文本流畅性（困惑度）
    inputs = tokenizer(text, return_tensors="pt").to(device)  # 将输入移动到MPS设备
    loss = model(**inputs, labels=inputs["input_ids"]).loss
    perplexity = torch.exp(loss).item()

    # 2. 计算文本情感
    sentiment_inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    sentiment_outputs = sentiment_model(**sentiment_inputs)
    sentiment_score = torch.softmax(sentiment_outputs.logits, dim=-1).mean().item()

    # 3. 计算句子嵌入相似性
    sentences = text.split(". ")
    sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)
    avg_cosine_similarity = cos_sim.mean().item()  # 平均相似度，衡量句子间一致性

    results.append({
        "File Name": os.path.basename(file_path),
        "Perplexity": perplexity,
        "Sentiment Score": sentiment_score,
        "Avg Sentence Similarity": avg_cosine_similarity
    })

# 将结果写入CSV文件
with open(output_csv_path, mode='w', newline='') as csv_file:
    fieldnames = ["File Name", "Perplexity", "Sentiment Score", "Avg Sentence Similarity"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)

print(f"Model-based writing quality assessment results saved to {output_csv_path}")