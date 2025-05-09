相似度評估.py
This Python script evaluates the semantic similarity between sentences from two versions of a transcript. It uses the BERT model to calculate semantic similarity and outputs the results as an Excel file.

Features
Extracts sentences from .docx files for analysis.
Filters out filler words and short sentences.
Computes semantic similarity using BERT and cosine similarity.
Outputs results into an Excel file with semantic similarity scores.
Requirements
Make sure you have the following Python libraries installed:

pandas
python-docx
transformers
torch
numpy
scikit-learn
Install them using:

bash
pip install pandas python-docx transformers torch numpy scikit-learn
Input Files
受訪者L_20250309.docx: The raw transcript.
受訪者L_20250309_整理版.docx: The cleaned-up transcript.
Output
The script generates an Excel file named 語意比對_高效率版_說話人2.xlsx, which contains the following columns:

最相近處理後段落編號: Edited paragraph number.
處理後內容: Content of the edited paragraph.
原始段落編號: Original paragraph number.
原始內容: Content of the original paragraph.
語意保持度: Semantic similarity score.
語意差異度: Semantic difference score.
How to Use
Place the input .docx files (受訪者L_20250309.docx and 受訪者L_20250309_整理版.docx) in the same directory as the script.
Run the script:
bash
python 相似度評估.py
The script will generate the output Excel file in the same directory.
Example
Here’s an example of running the script:

bash
python 相似度評估.py
The following message will appear upon completion:

Code
完成語意比對，結果已儲存為：語意比對_高效率版_說話人2.xlsx
Notes
Make sure your .docx files are properly formatted (e.g., sentences starting with 說話人2：).
Adjust the BERT model or tokenizer if you need different language support.
Commit the README File:
Once you’re done writing the README.md, scroll down and add a commit message (e.g., "Add README for 相似度評估.py").
Click the Commit new file button.
Let me know if you need help with any of these steps!
