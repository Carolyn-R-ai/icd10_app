import re
import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import google.generativeai as genai

st.set_page_config(
    page_title="ICD-10 Diagnosis Mapper",
    layout="centered",
    page_icon="🧬"
)

st.title("🧬 ICD-10 Matcher")

genai.configure(api_key="GEMINI_API_KEY")

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", str(text).lower()).strip()

@st.cache_data
def load_data():
    path = "icd10_with_diagnosis.xlsx"
    df = pd.read_excel(path)

    required = ["ICD10_Code", "Diagnosis_Name", "WHO_Full_Desc", "ICD10_Block"]
    for col in required:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            return pd.DataFrame()

    df = df.dropna(subset=["Diagnosis_Name", "ICD10_Block"])
    df["Diagnosis_Name"] = df["Diagnosis_Name"].astype(str)
    df["diag_clean"] = df["Diagnosis_Name"].apply(normalize)

    return df

def extract_focus_term(user_input):
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
You are a medical coding expert.

TASK:
Extract ONLY the most important organ or anatomical location
needed to map ICD-10.

RULES:
- Remove generic words like cancer, tumor, malignant, carcinoma, neoplasm.
- Return ONLY the key term in lowercase.
- No explanations.

INPUT:
"{user_input}"
"""

    try:
        resp = model.generate_content(prompt, generation_config={"temperature": 0})
        term = resp.text.strip().lower()
        return term if term else normalize(user_input)
    except:
        return normalize(user_input)

def gemini_match(user_input, candidates):
    model = genai.GenerativeModel("gemini-2.5-flash")

    options = "\n".join(f"- {c}" for c in candidates[:30])

    prompt = f"""
You are an ICD-10 medical coding specialist.

TASK:
Select the BEST matching diagnosis from the list.

RULES:
- Same organ + same disease only
- Return EXACT string from list
- If unsure, return NONE

INPUT:
"{user_input}"

LIST:
{options}
"""

    try:
        resp = model.generate_content(prompt, generation_config={"temperature": 0})
        ans = resp.text.strip()

        if ans != "NONE":
            for c in candidates:
                if c.lower() == ans.lower():
                    return c
    except:
        pass

    return None

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

    
@st.cache_data(show_spinner=False)
def cached_focus_term(text):
    return extract_focus_term(text)


def map_icd10(user_input, df):
    focus = cached_focus_term(user_input)

    exact = df[df["diag_clean"] == focus]
    if not exact.empty:
        return exact.iloc[0], "Exact Match", 100

    fuzzy = process.extractOne(
        focus, df["diag_clean"], scorer=fuzz.token_set_ratio
    )
    if fuzzy and fuzzy[1] >= 85:
        row = df.iloc[fuzzy[2]]
        return row, "Fuzzy Match", fuzzy[1]

    candidates = df["Diagnosis_Name"].tolist()[:30]
    gem = gemini_match(user_input, candidates)
    if gem:
        row = df[df["Diagnosis_Name"] == gem].iloc[0]
        return row, "Gemini AI Match", 95

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

            focus_term = extract_focus_term(query)
            top5 = get_top_probable_diagnoses(
                df, focus_term, result["ICD10_Block"]
            )

            st.markdown("Top 5 High-Probability Related Diagnoses")
            for i, (name, code, score) in enumerate(top5, start=1):
                st.write(f"{i}. **{name}** — `{code}`")


        else:
            st.error("Not found")

else:

    st.error("Dataset not loaded")




