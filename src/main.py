import streamlit as st
import torch
from transformers import pipeline


def analyze_text(text_to_analyze, min_length, max_length):
    summarizer = pipeline(
        "summarization",
        "pszemraj/long-t5-tglobal-base-16384-book-summary",
        device="cuda" if torch.cuda.is_available() else "cpu",
        min_length=min_length,
        max_length=max_length
    )

    result = summarizer(text_to_analyze)
    output = result[0]["summary_text"]
    return output


def main():
    st.title("Text summarization application")

    st.write("This is an application that analyses text and return its summary.")
    st.write("Based on \"long-t5-tglobal-base-16384 + BookSum\" model.")
    st.write("Model source: https://huggingface.co/pszemraj/long-t5-tglobal-base-16384-book-summary")

    sample = ""
    with open('sample.txt') as f:
        lines = f.readlines()
        for line in lines:
            sample += line

    with st.form("my_form"):
        txt = st.text_area("Enter your text here:", sample)
        summary_min_length = st.slider("You may set the minimum length of the summary:",
                                       min_value=1, max_value=100, value=8)
        summary_max_length = st.slider("You may set the maximum length of the summary:",
                                       min_value=10, max_value=1024, value=512)

        submitted = st.form_submit_button("Submit")
        if submitted:
            if txt == "":
                st.error("No text to analyze")
            elif summary_min_length > summary_max_length:
                st.error("Minimal length cannot be greater than maximum length")
            else:
                result = analyze_text(txt, summary_min_length, summary_max_length)

                st.write("**Result:**")
                st.write(result)


if __name__ == '__main__':
    main()
