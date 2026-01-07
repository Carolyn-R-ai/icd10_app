import pandas as pd
from rapidfuzz import process, fuzz
from openai import OpenAI
import re
import os

import streamlit as st

st.set_page_config(
    page_title="ICD-10 App Matcher",
    layout="centered"
)

st.title("ICD-10 App Matcher")

df = pd.read_excel('Diagnoses.xlsx')

df['Source'] = df['Source'].astype(str).str.strip()
df['MappingFieldValue'] = df['MappingFieldValue'].astype(str).str.strip()
df['ICD10 Code'] = df['ICD10 Code'].astype(str).str.strip()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def normalize_diagnosis_llm(text):
    choices = df['MappingFieldValue'].dropna().unique()
    prompt = f"""
    Normalize this diagnosis: "{text}"
    Only return one of these standard names exactly as written:
    {', '.join(choices)}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def map_input(text, df, threshold=90):

    text_lower = text.lower().strip()
    text_lower = re.sub(r'[^a-z0-9\s]', '', text_lower)

    row = df[df['Source'].str.lower() == text_lower]
    if not row.empty:
        return row['MappingFieldValue'].iloc[0], row['ICD10 Code'].iloc[0], 100

    row = df[df['MappingFieldValue'].str.lower() == text_lower]
    if not row.empty:
        return row['MappingFieldValue'].iloc[0], row['ICD10 Code'].iloc[0], 100

    row = df[df['ICD10 Code'].str.lower() == text_lower]
    if not row.empty:
        return row['MappingFieldValue'].iloc[0], row['ICD10 Code'].iloc[0], 100

    return None, None, 0

def predict_icd10_llm_assisted(text):
    normalized = normalize_diagnosis_llm(text)
    mapping_value, icd_code, confidence = map_input(normalized, df)
    return {
        "Input": text,
        "MappingFieldValue": mapping_value,
        "ICD10 Code": icd_code
    }
    
user_input = st.text_input("Enter a Diagnosis")

if st.button("Enter"):
    if user_input.strip() == "":
        st.warning("Please enter a value.")
    else:
        result = predict_icd10_llm_assisted(user_input)

        if result["ICD10 Code"]:
            st.write("**Input:**", result["Input"])
            st.write("**Mapping Field Value:**", result["MappingFieldValue"])
            st.write("**ICD-10 Code:**", result["ICD10 Code"])
        else:
            st.error("No match found")
