import streamlit as st
import requests

def main():
    page_icon="\U0001F916"
    st.set_page_config(
        page_title="NER", page_icon=page_icon, initial_sidebar_state="expanded"
    )
    text = st.text_area("Enter your text")
    button = st.button("rcognize")

    if button:
        response = requests.post("http://127.0.0.1:8000/ner",json={"name":text})
        if response.status_code>=400: st.error(response.text);return
        st.write(response.json()["lables"],response.json()["tokens"])
if __name__ == "__main__":main()