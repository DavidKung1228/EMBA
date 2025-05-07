import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import torch
import re
import time
import os
import pickle
import random

# 載入SentenceTransformer模型
print("[INFO] 載入 SentenceTransformer 模型...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("[INFO] 模型載入完成。\n")

def load_keywords_from_files(file_paths, cache_file="keywords_cache.pkl"):
    """從Excel檔案中載入關鍵詞，使用快取加速"""
    # 檢查是否有快取檔案
    if os.path.exists(cache_file):
        print(f"[INFO] 發現快取檔案，嘗試載入...")
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 檢查快取的檔案清單是否與當前相同
            if set(cache_data["file_paths"]) == set(file_paths):
                print(f"[INFO] 成功從快取載入 {len(cache_data['all_keywords'])} 個關鍵詞。")
                return cache_data["all_keywords"], cache_data["keyword_to_themes"], cache_data["keyword_counts"]
            else:
                print(f"[INFO] 快取檔案與當前檔案清單不符，重新處理...")
        except Exception as e:
            print(f"[WARN] 載入快取時出錯: {e}，將重新處理檔案...")
    
    # 如果沒有快取或快取無效，則重新處理
    all_keywords = []
    keyword_to_themes = defaultdict(set)
    keyword_counts = defaultdict(int)
    
    for file_path in file_paths:
        print(f"[INFO] 開始處理文件: {file_path}")
        try:
            df = pd.read_excel(file_path)
            if '關鍵字' in df.columns:
                for idx, row in df.iterrows():
                    if pd.notna(row['關鍵字']):
                        # 拆分關鍵詞 (使用","、"、";"等分隔符)
                        keywords = re.split(r"[,，;；/、\\n]", str(row['關鍵字']))
                        for keyword in keywords:
                            keyword = keyword.strip()
                            if keyword:
                                all_keywords.append(keyword)
                                keyword_counts[keyword] += 1
                                # 確保使用範疇欄位
                                for field in ['範疇分類', '範疇', '主題範疇', '分類']:
                                    if field in df.columns and pd.notna(row.get(field)):
                                        keyword_to_themes[keyword].add(str(row[field]).strip())
                                        break
            print(f"[INFO] 完成 {file_path}，共載入 {len(df)} 行資料。")
        except Exception as e:
            print(f"[ERROR] 處理文件 {file_path} 時出錯: {e}")
    
    print(f"[INFO] 總共載入 {len(all_keywords)} 個關鍵詞。\n")
    
    # 保存快取
    cache_data = {
        "file_paths": file_paths,
        "all_keywords": all_keywords,
        "keyword_to_themes": keyword_to_themes,
        "keyword_counts": keyword_counts
    }
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"[INFO] 關鍵詞資料已快取到 {cache_file}")
    except Exception as e:
        print(f"[WARN] 保存快取時出錯: {e}")
    
    return all_keywords, keyword_to_themes, keyword_counts

def analyze_keyword_relatedness(target_word, keyword_list, model, threshold_medium=0.6, batch_size=50, 
                               cache_dir="similarity_cache"):
    """分析關鍵詞與目標詞彙的相關性 (批處理版本)，使用快取加速"""
    
    # 確保快取目錄存在
    os.makedirs(cache_dir, exist_ok=True)
    
    # 檢查是否有此目標詞彙的快取
    cache_file = f"{cache_dir}/{target_word}_similarity.pkl"
    if os.path.exists(cache_file):
        print(f"[INFO] 發現 '{target_word}' 的相似度快取，嘗試載入...")
        try:
            with open(cache_file, 'rb') as f:
                cached_results = pickle.load(f)
            
            # 簡單檢查快取是否有效
            if cached_results and isinstance(cached_results, list) and "關鍵詞" in cached_results[0]:
                print(f"[INFO] 成功從快取載入 {len(cached_results)} 個相關詞彙。")
                return cached_results
        except Exception as e:
            print(f"[WARN] 載入相似度快取時出錯: {e}，將重新計算...")
    
    print(f"[INFO] 分析關鍵詞與 '{target_word}' 的相關性 (共 {len(keyword_list)} 個關鍵詞)...")
    
    results = []
    target_embedding = model.encode(target_word, convert_to_tensor=True)
    
    # 批處理計算相似度
    for i in range(0, len(keyword_list), batch_size):
        batch = keyword_list[i:i+batch_size]
        print(f"[INFO] 處理批次 {i//batch_size + 1}/{(len(keyword_list)-1)//batch_size + 1} ({i+1}-{min(i+batch_size, len(keyword_list))}/{len(keyword_list)})")
        
        # 計算批次嵌入
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        
        # 計算與目標詞的相似度
        similarities = util.cos_sim(target_embedding, batch_embeddings)[0]
        
        # 處理結果
        for j, similarity in enumerate(similarities):
            similarity_value = similarity.item()
            if similarity_value >= threshold_medium:
                relation_level = "高度相關" if similarity_value >= 0.8 else "中度相關"
                results.append({
                    "關鍵詞": batch[j],
                    "相關程度": relation_level,
                    "相似度分數": round(similarity_value, 4)
                })
    
    results = sorted(results, key=lambda x: x["相似度分數"], reverse=True)
    print(f"[INFO] 找到 {len(results)} 個與 '{target_word}' 中度以上相關的詞彙。\n")
    
    # 保存快取
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"[INFO] 相似度結果已快取到 {cache_file}")
    except Exception as e:
        print(f"[WARN] 保存相似度快取時出錯: {e}")
    
    return results

def analyze_semantic_keywords(file_paths, target_word, relation_threshold=0.6, batch_size=50, use_cache=True):
    """分析關鍵詞並統計語意組頻率"""
    print(f"[INFO] 開始分析目標詞彙 '{target_word}' 的相關性...\n")
    start_time = time.time()
    
    # 載入關鍵詞（使用快取）
    all_keywords, keyword_to_themes, keyword_counts = load_keywords_from_files(
        file_paths, 
        cache_file="keywords_cache.pkl" if use_cache else None
    )
    unique_keywords = list(set(all_keywords))
    print(f"[INFO] 去重後共有 {len(unique_keywords)} 個獨特關鍵詞。\n")
    
    # 分析關鍵詞相關性（使用快取）
    relatedness_results = analyze_keyword_relatedness(
        target_word, 
        unique_keywords, 
        model, 
        relation_threshold, 
        batch_size,
        cache_dir="similarity_cache" if use_cache else None
    )
    relatedness_df = pd.DataFrame(relatedness_results)
    
    if not relatedness_df.empty:
        # 添加主題範圍資訊
        themes_list = []
        for row in relatedness_df.itertuples():
            keyword = row.關鍵詞
            themes = keyword_to_themes.get(keyword, set())
            themes_str = ", ".join(sorted(themes)) if themes else "未分類"
            themes_list.append(themes_str)
        
        relatedness_df["來源主題範圍"] = themes_list
        relatedness_df["出現次數"] = relatedness_df["關鍵詞"].apply(lambda k: keyword_counts.get(k, 0))
    
    end_time = time.time()
    print(f"[INFO] 分析完成，耗時 {end_time - start_time:.2f} 秒。")
    return relatedness_df

def save_excel_with_retry(df, file_path, max_retries=5, retry_delay=1):
    """保存Excel檔案，遇到錯誤時重試"""
    for attempt in range(max_retries):
        try:
            # 如果文件存在，則先嘗試使用臨時文件名稱
            if os.path.exists(file_path) and attempt > 0:
                temp_file = f"{os.path.splitext(file_path)[0]}_{random.randint(1000, 9999)}.xlsx"
                df.to_excel(temp_file, index=False)
                print(f"[INFO] 已保存到臨時檔案: {temp_file}")
                
                # 嘗試刪除原始檔案並重命名
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    os.rename(temp_file, file_path)
                    print(f"[INFO] 臨時檔案已重命名為: {file_path}")
                    return True
                except Exception as e:
                    print(f"[WARN] 無法重命名檔案: {e}")
                    print(f"[INFO] 結果已保存到: {temp_file}")
                    return True
            else:
                # 直接嘗試保存
                df.to_excel(file_path, index=False)
                print(f"[INFO] 結果已保存到: {file_path}")
                return True
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"[WARN] 保存到 {file_path} 時出現權限錯誤，嘗試重試 ({attempt+1}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                # 最後一次嘗試，改用時間戳文件名
                fallback_file = f"{os.path.splitext(file_path)[0]}_{int(time.time())}.xlsx"
                print(f"[WARN] 無法保存到 {file_path}，嘗試使用替代檔案名: {fallback_file}")
                try:
                    df.to_excel(fallback_file, index=False)
                    print(f"[INFO] 結果已保存到替代檔案: {fallback_file}")
                    return True
                except Exception as sub_e:
                    print(f"[ERROR] 保存到替代檔案時出錯: {sub_e}")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[WARN] 保存到 {file_path} 時出錯: {e}，嘗試重試 ({attempt+1}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                print(f"[ERROR] 保存到 {file_path} 失敗: {e}")
    
    return False

def process_multiple_targets(file_paths, target_words, relation_threshold=0.6, batch_size=50, 
                            output_dir="結果", use_cache=True):
    """處理多個目標詞彙"""
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 建立整合結果的DataFrame
    all_results = pd.DataFrame()
    
    # 儲存每個目標詞彙的相關詞集合
    target_to_keywords = {}
    
    # 處理每個目標詞彙
    for target_word in target_words:
        print(f"\n[INFO] 開始處理目標詞彙: {target_word}")
        result_df = analyze_semantic_keywords(
            file_paths,
            target_word=target_word,
            relation_threshold=relation_threshold,
            batch_size=batch_size,
            use_cache=use_cache
        )
        
        if not result_df.empty:
            # 添加目標詞彙欄位
            result_df["目標詞彙"] = target_word
            
            # 保存單個結果
            output_file = f"{output_dir}/{target_word}_相關詞分析結果.xlsx"
            success = save_excel_with_retry(result_df, output_file)
            if success:
                print(f"[INFO] 結果已另存為 '{output_file}'")
            
            # 儲存相關詞集合
            target_to_keywords[target_word] = set(result_df["關鍵詞"])
            
            # 整合到總結果
            all_results = pd.concat([all_results, result_df], ignore_index=True)
        else:
            print(f"[WARN] 目標詞彙 '{target_word}' 未找到相關結果。")
    
    # 保存整合結果
    if not all_results.empty:
        # 修改合併結果的格式，按照目標詞彙合併
        aggregated_results = []
        
        for target_word in target_words:
            if target_word in target_to_keywords:
                # 篩選出當前目標詞彙的所有相關詞
                related_keywords = target_to_keywords[target_word]
                
                # 從all_results中獲取這些相關詞的詳細資訊
                keywords_info = all_results[
                    (all_results["目標詞彙"] == target_word)
                ].copy()
                
                if not keywords_info.empty:
                    # 計算總出現次數
                    total_count = keywords_info["出現次數"].sum()
                    
                    # 收集所有主題範圍
                    all_themes = set()
                    for themes in keywords_info["來源主題範圍"]:
                        if pd.notna(themes) and themes != "未分類":
                            all_themes.update([t.strip() for t in themes.split(',')])
                    
                    # 生成格式化的字符串
                    keywords_str = "{" + ", ".join(related_keywords) + "}"
                    themes_str = "{" + ", ".join(all_themes) + "}" if all_themes else "未分類"
                    
                    # 添加至結果
                    aggregated_results.append({
                        "目標詞彙": target_word,
                        "相關關鍵詞": keywords_str,
                        "相關詞數量": len(related_keywords),
                        "總出現次數": total_count,
                        "來源主題範圍": themes_str
                    })
        
        # 創建合併後的DataFrame
        merged_df = pd.DataFrame(aggregated_results)
        
        # 按相關詞數量排序
        merged_df = merged_df.sort_values(by=["相關詞數量", "總出現次數"], ascending=False)
        
        # 保存新格式的合併結果
        save_excel_with_retry(merged_df, f"{output_dir}/新合併關鍵詞分析結果.xlsx")
        
        # 同時保存原始完整結果以備不時之需
        save_excel_with_retry(all_results, f"{output_dir}/全部目標詞彙分析結果.xlsx")
        
        # 合併處理: 對每個關鍵詞，合併其所有相關的目標詞彙 (原始合併方式)
        keyword_based_results = all_results.groupby('關鍵詞').agg({
            '相關程度': lambda x: max(x),  # 取最高相關程度
            '相似度分數': lambda x: max(x),  # 取最高相似度分數
            '來源主題範圍': lambda x: ', '.join(set(filter(None, x))),  # 合併主題範圍
            '出現次數': lambda x: x.iloc[0],  # 保持原有出現次數
            '目標詞彙': lambda x: ', '.join(set(x))  # 合併目標詞彙
        }).reset_index()
        
        # 將"目標詞彙"欄位的數量統計為新的"目標詞彙出現次數"
        keyword_based_results['目標詞彙出現次數'] = keyword_based_results['目標詞彙'].apply(lambda x: len(x.split(', ')))
        
        # 將目標詞彙欄位改名為"相關目標詞彙"
        keyword_based_results = keyword_based_results.rename(columns={'目標詞彙': '相關目標詞彙'})
        
        # 按目標詞彙出現次數排序
        keyword_based_results = keyword_based_results.sort_values(by=['目標詞彙出現次數', '出現次數'], ascending=False)
        
        # 保存原有格式的合併結果
        save_excel_with_retry(keyword_based_results, f"{output_dir}/合併關鍵詞分析結果.xlsx")
    
    return all_results, merged_df if 'merged_df' in locals() else None

if __name__ == "__main__":
    file_paths = [
        "受訪者F_20250322_初期編碼完成版.xlsx",
        "受訪者I_20250326_初期編碼完成版..xlsx",
        "受訪者J_20250408_初期編碼完成版.xlsx",
        "受訪者L_20250309_初期編碼完成版.xlsx",
        "受訪者T_20250412_初期編碼完成版.xlsx"
    ]
    
    # 多個目標詞彙
    target_words = [
        "線上課表", "顧客關係", "品牌語言", "社群經營", "基礎建設", "語言模組", 
        "人力縮減", "疫情影響", "即時溝通", "族群辨識", "顧客信任", "教練不足", 
        "世壯運", "家庭挑戰", "成本控管", "收入來源", "社群平台", "角色轉換", 
        "小朋友品牌建立", "人事成本", "訓練營", "鐵人三項", "競爭者分析", 
        "低成本試營運", "直播互動", "標準化", "文化風格", "游泳池", "租金負擔", 
        "跑班持續", "離職選擇", "情緒管理", "未來展望", "腳踏車品牌", "口令簡化", 
        "營養品", "運動動機", "臉書平台", "影響力雙向", "學生優惠", "粉絲互動", 
        "信任關係", "出版動機", "APP系統", "資料庫建構", "身體改變", "穩定性不足", 
        "LINE互動", "功率課", "EMBA", "大陸拓展", "單一場地", "服裝規範模糊", 
        "自主工作", "怕被笑", "活動光感", "手錶抽獎", "影音轉型", "沒人來", 
        "策略不變", "觀光結合", "交通便利性", "政策冷感", "成本壓力", "用戶觀察", 
        "負債經驗", "社會需求"
    ]
    
    # 設定參數
    relation_threshold = 0.6
    batch_size = 50
    output_dir = "關鍵詞相關性分析結果"
    use_cache = True  # 是否使用快取功能
    
    # 分析多個目標詞彙
    all_results, merged_results = process_multiple_targets(
        file_paths,
        target_words,
        relation_threshold=relation_threshold,
        batch_size=batch_size,
        output_dir=output_dir,
        use_cache=use_cache
    )
    
    print("\n[INFO] 所有目標詞彙分析完成！")
    if merged_results is not None:
        print(f"[INFO] 相關詞分組總數: {len(merged_results)}")
    print(f"[INFO] 目標詞彙總數: {len(target_words)}")
