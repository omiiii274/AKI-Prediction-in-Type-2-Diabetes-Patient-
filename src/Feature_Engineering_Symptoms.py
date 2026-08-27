import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

INPUT_PATH = "data/processed/ed_symptom_features.parquet"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_parquet(INPUT_PATH)

tfidf = TfidfVectorizer(max_features=60, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['chiefcomplaint'])
X = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"symptom_{w}" for w in tfidf.get_feature_names_out()])

# Add explicit keyword flags for stronger, cleaner signal (as discussed)
keyword_flags = {
    'has_pee_symptom': df['chiefcomplaint'].str.contains('urin|pee|oliguria|anuria', case=False, na=False).astype(int),
    'has_swelling': df['chiefcomplaint'].str.contains('swell|edema', case=False, na=False).astype(int),
    'has_breathless': df['chiefcomplaint'].str.contains('breath|dyspnea|sob', case=False, na=False).astype(int),
    'has_confusion': df['chiefcomplaint'].str.contains('confus|altered|lethar|sleepy|drowsy', case=False, na=False).astype(int),
    'has_nausea': df['chiefcomplaint'].str.contains('nausea|vomit', case=False, na=False).astype(int),
    'has_thirst': df['chiefcomplaint'].str.contains('thirst|dehydrat', case=False, na=False).astype(int),
}
keyword_df = pd.DataFrame(keyword_flags)

X = pd.concat([keyword_df.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
y = df['ever_aki'].reset_index(drop=True)

print(f"[INFO] Final feature matrix: {X.shape}")
print(f"[INFO] Columns: {list(X.columns)[:10]}...")

joblib.dump(tfidf, os.path.join(OUTPUT_DIR, "symptom_tfidf_vectorizer.joblib"))
X.to_parquet("data/processed/symptom_model_features.parquet", index=False)
y.to_frame('ever_aki').to_parquet("data/processed/symptom_model_labels.parquet", index=False)

print("[SUCCESS] Vitals-free symptom features rebuilt.")