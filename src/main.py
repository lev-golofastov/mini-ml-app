import streamlit as st
import torch
from transformers import pipeline


def analyze_text(text_to_analyze):
	summarizer = pipeline(
		"summarization",
		"pszemraj/long-t5-tglobal-base-16384-book-summary",
		device=0 if torch.cuda.is_available() else -1,
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

	st.write("A sample text for testing, found on model\'s web page:")
	st.write(sample)

	with st.form("my_form"):
		txt = st.text_area("Enter your text here:")

		submitted = st.form_submit_button("Submit")
		if submitted:
			if txt == "":
				st.error("No text to analyze")
			else:
				result = analyze_text(txt)

				st.write("**Result:**")
				st.write(result)


if __name__ == '__main__':
	main()
