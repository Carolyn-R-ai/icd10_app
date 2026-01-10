import re
import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz

st.set_page_config(
    page_title="ICD-10 Matcher", 
    layout="centered"
)

st.title("ICD-10 Matcher")

def normalize(text: str) -> str:
    """Lowercase and remove special characters."""
    return re.sub(r"[^a-z0-9\s]", "", str(text).lower()).strip()

def extract_focus_term_offline(user_input):
    """Remove generic words like cancer, tumor, malignant, carcinoma, neoplasm."""
    words_to_remove = ["cancer", "tumor", "malignant", "carcinoma", "neoplasm", "lesion", "disease", "syndrome"]
    clean = ' '.join([w for w in user_input.lower().split() if w not in words_to_remove])
    return clean.strip()

@st.cache_data
def load_data(file_path="icd10_with_diagnosis.xlsx"):
    """Load ICD-10 dataset."""
    try:
        df = pd.read_excel(file_path)
        required_cols = ["ICD10_Code", "Diagnosis_Name", "WHO_Full_Desc", "ICD10_Block"]
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing column: {col}")
                return pd.DataFrame()
        df = df.dropna(subset=["Diagnosis_Name", "ICD10_Block"])
        df["Diagnosis_Name"] = df["Diagnosis_Name"].astype(str)
        df["diag_clean"] = df["Diagnosis_Name"].apply(normalize)
        return df
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return pd.DataFrame()

def get_top_probable_diagnoses(df, focus_term, icd_block, limit=5):
    """
    Return top probable diagnoses with ICD-10 codes.
    Priority: same ICD10_Block → highest fuzzy score
    """
    block_df = df[df["ICD10_Block"] == icd_block]
    if len(block_df) < limit:
        block_df = df

    scored = []
    for _, row in block_df.iterrows():
        score = fuzz.token_set_ratio(focus_term, row["diag_clean"])
        scored.append((
            row["Diagnosis_Name"],
            row["ICD10_Code"],
            score
        ))
    scored = sorted(scored, key=lambda x: x[2], reverse=True)
    return scored[:limit]

def map_icd10(user_input, df):
    focus = extract_focus_term_offline(user_input)

    exact = df[df["diag_clean"] == focus]
    if not exact.empty:
        return exact.iloc[0], "Exact Match", 100

    fuzzy = process.extractOne(focus, df["diag_clean"], scorer=fuzz.token_set_ratio)
    if fuzzy and fuzzy[1] >= 85:
        row = df.iloc[fuzzy[2]]
        return row, "Fuzzy Match", fuzzy[1]

    candidates = df["Diagnosis_Name"].tolist()
    return None, "Not Found", 0

df = load_data()

if not df.empty:
    query = st.text_input("Enter Diagnosis")

    if st.button("Search ICD-10") and query.strip():
        result, method, confidence = map_icd10(query, df)

        if result is not None:
            st.success(f"### Match Found ({method})")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("ICD-10 Code", result["ICD10_Code"])
            with col2:
                st.metric("Confidence", f"{confidence}%")

            st.write(f"**Diagnosis:** {result['Diagnosis_Name']}")
            st.write(f"**ICD-10 Block:** {result['ICD10_Block']}")
            st.info(result["WHO_Full_Desc"])

            focus_term = extract_focus_term_offline(query)
            top5 = get_top_probable_diagnoses(df, focus_term, result["ICD10_Block"])

            st.markdown("### Top 5 High-Probability Related Diagnoses")
            for i, (name, code, score) in enumerate(top5, start=1):
                st.write(f"{i}. **{name}** — `{code}`")
        else:
            st.error("No suitable ICD-10 match found. Please try a different term.")
else:
    st.error("Dataset not loaded. Check the Excel file path.")
