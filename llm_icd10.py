from openai import OpenAI
import streamlit as st

st.set_page_config(
    page_title="ICD-10 App Matcher",
    layout="centered"
)

st.title("ICD-10 App Matcher")

user_input = st.text_input("Enter a Diagnosis")

client = OpenAI(api_key="OPENAI_API_KEY")
def get_icd10_cancer(user_input):
    prompt = f"""
You are an expert in ICD-10 cancer codes.

Map the following cancer name or description to its correct ICD-10 code and official name.
If unsure, return 'Not Found'.

User Input: "{user_input}"

Output format:
ICD-10 Code - Official Name
"""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",   
        messages=[{"role": "user", "content": prompt}],
        temperature=0  
    )
    
   
    answer = response.choices[0].message.content.strip()
    return answer

if user_input:
    result = get_icd10_cancer(user_input)
    st.success(f" Result: {result}")
