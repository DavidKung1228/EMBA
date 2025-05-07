1 Model and data initialization


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') # semantic embedding model
2 Keyword reading and standardization


def load_keywords_from_files(file_paths):
Read multiple Excel files, extract keyword fields, and handle common delimiters
for file in file_paths:
df = pd.read_excel(file)
for row in df.itertuples():
keywords = re.split(r'[,，;；/、\\n]', str(row.keywords))
3 Keyword semantic comparison (embedding + similarity)

def analyze_keyword_relatedness(target_word, keyword_list):
target_vec = model.encode(target_word, convert_to_tensor=True)
batch_vec = model.encode(keyword_list, convert_to_tensor=True)
similarities = util.cos_sim(target_vec, batch_vec)[0]
4 Similarity filtering and hierarchical classification

for keyword, score in zip(keyword_list, similarities):
if score > 0.6:
level = "highly correlated" if score > 0.8 else "moderately correlated"
5 Export analysis results

df.to_excel("Keyword correlation analysis results.xlsx", index=False)
