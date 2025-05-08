import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sentence_transformers import SentenceTransformer, util

# 資料載入
original_df = pd.read_excel('Original_Text.xlsx')
human_df = pd.read_excel('F_Human_Coding.xlsx')
ai_df = pd.read_excel('F_AI_Coding.xlsx')
print("original_df 欄位:", original_df.columns.tolist())
print("human_df 欄位:", human_df.columns.tolist())
print("ai_df 欄位:", ai_df.columns.tolist())

# 明確重新命名欄位避免重複
human_df.rename(columns={'原文段落':'原文段落_human','初期編碼':'初期編碼_human'}, inplace=True)
ai_df.rename(columns={'原文段落':'原文段落_ai','初期編碼':'初期編碼_ai'}, inplace=True)

# 合併資料
merged_df = original_df.merge(
    human_df[['Index', '原文段落_human', '初期編碼_human']], on='Index', how='left'
).merge(
    ai_df[['Index', '原文段落_ai', '初期編碼_ai']], on='Index', how='left'
)

# 載入模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 語意相似度函數
def semantic_similarity(text1, text2):
    if pd.isna(text1) or pd.isna(text2):
        return 0
    embeddings = model.encode([text1, text2])
    return util.cos_sim(embeddings[0], embeddings[1]).item()

# 主比對迴圈 (確保TN值一定出現)
results = []
for idx, row in merged_df.iterrows():
    ori_text = row['原文段落']
    ai_text = row['原文段落_ai']
    human_text = row['原文段落_human']
    sim_ori_ai = semantic_similarity(ori_text, ai_text)
    sim_ori_human = semantic_similarity(ori_text, human_text)
    coding_sim = None  # 預先指定預設值，避免未定義問題
    
    if (sim_ori_ai < 0.5) and (sim_ori_human < 0.5):
        result = 'TN'
    elif (sim_ori_ai >= 0.5) and (sim_ori_human < 0.5):
        result = 'FP'
        print(f"[FP Warning] Human相似度: {sim_ori_human:.2f}\n原文:{ori_text}\nHuman:{human_text}\n")
    elif (sim_ori_ai < 0.5) and (sim_ori_human >= 0.5):
        result = 'FN'
        print(f"[FN Warning] AI相似度: {sim_ori_ai:.2f}\n原文:{ori_text}\nAI:{ai_text}\n")
    else:
        ai_coding = row['初期編碼_ai']
        human_coding = row['初期編碼_human']
        ai_has_coding = int(pd.notna(ai_coding))
        human_has_coding = int(pd.notna(human_coding))
        
        if ai_has_coding == 0 and human_has_coding == 0:
            result = 'TN'
        elif ai_has_coding == 1 and human_has_coding == 0:
            result = 'FP'
        elif ai_has_coding == 0 and human_has_coding == 1:
            result = 'FN'
        else:
            coding_sim = semantic_similarity(str(ai_coding), str(human_coding))
            result = 'TP' if coding_sim >= 0.5 else 'FP'
    
    # 每次迴圈一定要記錄coding_sim (沒有比對則為None)
    results.append({
        'Index': row['Index'],
        '原文段落': ori_text,
        '原文AI比對值': sim_ori_ai,
        '原文human比對值': sim_ori_human,
        '初期編碼_ai': row['初期編碼_ai'],
        '初期編碼_human': row['初期編碼_human'],
        '初期編碼_比對值': coding_sim if coding_sim is not None else 0,
        'result': result
    })

# 建立DataFrame
final_df = pd.DataFrame(results)

# 調整輸出欄位順序
final_df = final_df[['Index', '原文段落', '原文AI比對值', '原文human比對值',
                     '初期編碼_ai', '初期編碼_human', '初期編碼_比對值', 'result']]

# 統計結果
summary_counts = final_df['result'].value_counts().reset_index()
summary_counts.columns = ['Metric', 'Count']

# 計算評估指標
# 首先將類別轉換為二進制（TP/FP/FN/TN -> 1/0）
y_true = []
y_pred = []

for result in final_df['result']:
    if result == 'TP':
        y_true.append(1)
        y_pred.append(1)
    elif result == 'FP':
        y_true.append(0)
        y_pred.append(1)
    elif result == 'FN':
        y_true.append(1)
        y_pred.append(0)
    elif result == 'TN':
        y_true.append(0)
        y_pred.append(0)

# 從混淆矩陣計算指標
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# 直接計算 Precision 和 Recall (避免除以零的問題)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

# 計算 F1-score
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# 計算 Cohen's Kappa
kappa = cohen_kappa_score(y_true, y_pred)

# 計算 Balanced Accuracy
balanced_acc = balanced_accuracy_score(y_true, y_pred)

# 計算 Matthews 相關係數 (MCC)
mcc = matthews_corrcoef(y_true, y_pred)

# 創建評估指標的DataFrame
metrics_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-score', 'Cohen\'s Kappa', 'Balanced Accuracy', 'Matthews相關係數 (MCC)'],
    'Value': [precision, recall, f1, kappa, balanced_acc, mcc]
})

# 輸出Excel
output_file = 'Final_Result.xlsx'
with pd.ExcelWriter(output_file) as writer:
    final_df.to_excel(writer, sheet_name='Detailed_Comparison', index=False)
    summary_counts.to_excel(writer, sheet_name='Summary', index=False)
    metrics_df.to_excel(writer, sheet_name='Evaluation_Metrics', index=False)

print(f"✅ 完整比對完成，結果儲存至 {output_file}")
print("\n評估指標:")
for index, row in metrics_df.iterrows():
    print(f"{row['Metric']}: {row['Value']:.4f}")

# 計算總計值
total_samples = len(final_df)
print(f"\n總樣本數: {total_samples}")
print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
