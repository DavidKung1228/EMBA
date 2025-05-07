import pandas as pd
from docx import Document
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

# 初始化模型
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")
model.eval()

# 擷取說話人2語句
def extract_s2_text(doc_path):
    doc = Document(doc_path)
    return [p.text.replace("說話人2：", "").strip() for p in doc.paragraphs if p.text.startswith("說話人2：")]

# 篩選贅詞與短句
def is_valid(text):
    return len(text.strip()) >= 4 and text.strip() not in {"對", "嗯", "喔", "哦", "是", "嘿", "啊", "好"}

# 向量化
def get_vec(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# 讀取逐字稿
raw_s2 = extract_s2_text("受訪者L_20250309.docx")
edited_s2 = extract_s2_text("受訪者L_20250309_整理版.docx")

# 過濾與向量化
raw_s2_filtered = [(i, t) for i, t in enumerate(raw_s2) if is_valid(t)]
raw_vectors = [get_vec(t) for _, t in raw_s2_filtered]

edited_vectors = [get_vec(t) for t in edited_s2]

# 建立語意矩陣：處理後句為主，找原始最相近語句
sim_matrix = cosine_similarity(edited_vectors, raw_vectors)

results = []
used_raw_indices = set()

for j, row in enumerate(sim_matrix):
    sorted_indices = np.argsort(-row)  # 由大到小排序相似度
    for idx in sorted_indices:
        if idx not in used_raw_indices:
            best_score = row[idx]
            best_raw_idx, best_raw_text = raw_s2_filtered[idx]
            used_raw_indices.add(idx)
            results.append({
                "最相近處理後段落編號": j + 1,
                "處理後內容": edited_s2[j],
                "原始段落編號": best_raw_idx + 1,
                "原始內容": best_raw_text,
                "語意保持度": round(best_score, 4),
                "語意差異度": round(1 - best_score, 4)
            })
            break

# 輸出
df = pd.DataFrame(results)
df.to_excel("語意比對_高效率版_說話人2.xlsx", index=False)
print("完成語意比對，結果已儲存為：語意比對_高效率版_說話人2.xlsx")
