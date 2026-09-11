import pandas as pd  # I use pandas to load and manipulate the symptom dataset.
import numpy as np   # I use NumPy for numerical operations if needed during preprocessing.
from sklearn.feature_extraction.text import TfidfVectorizer  # I use TF-IDF to convert chief-complaint text into numeric features.
import joblib  # I use joblib to save the fitted vectoriser and other objects.
import os  # I use os to create output directories and construct file paths.


INPUT_PATH = "data/processed/ed_symptom_features.parquet"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# I load the pre-extracted ED chief-complaint records with AKI labels.
df = pd.read_parquet(INPUT_PATH)


# I limit vocabulary size and use unigrams + bigrams to capture common symptom phrases without overfitting.
tfidf = TfidfVectorizer(max_features=60, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['chiefcomplaint'])
X = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"symptom_{w}" for w in tfidf.get_feature_names_out()])


# Add explicit keyword flags for stronger, cleaner signal (as discussed)
# I add binary flags for clinically salient AKI-related terms to complement the TF-IDF features.
keyword_flags = {
    'has_pee_symptom': df['chiefcomplaint'].str.contains('urin|pee|oliguria|anuria', case=False, na=False).astype(int),
    'has_swelling': df['chiefcomplaint'].str.contains('swell|edema', case=False, na=False).astype(int),
    'has_breathless': df['chiefcomplaint'].str.contains('breath|dyspnea|sob', case=False, na=False).astype(int),
    'has_confusion': df['chiefcomplaint'].str.contains('confus|altered|lethar|sleepy|drowsy', case=False, na=False).astype(int),
    'has_nausea': df['chiefcomplaint'].str.contains('nausea|vomit', case=False, na=False).astype(int),
    'has_thirst': df['chiefcomplaint'].str.contains('thirst|dehydrat', case=False, na=False).astype(int),
}
keyword_df = pd.DataFrame(keyword_flags)


# I concatenate keyword flags and TF-IDF features into a single symptom-feature matrix.
X = pd.concat([keyword_df.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
y = df['ever_aki'].reset_index(drop=True)


print(f"[INFO] Final feature matrix: {X.shape}")
print(f"[INFO] Columns: {list(X.columns)[:10]}...")


# I save the fitted vectoriser so the deployed app can transform new symptom text identically.
joblib.dump(tfidf, os.path.join(OUTPUT_DIR, "symptom_tfidf_vectorizer.joblib"))
# I persist the feature matrix and labels for use in train_symptom_model.py.
X.to_parquet("data/processed/symptom_model_features.parquet", index=False)
y.to_frame('ever_aki').to_parquet("data/processed/symptom_model_labels.parquet", index=False)


print("[SUCCESS] Vitals-free symptom features rebuilt.")