import streamlit as st
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer
from transformers import pipeline

# Define the title of the app
st.title("NLP Mini Project")

# Add some text to the app
st.write("Question and Answering system")


model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
nlp = pipeline('question-answering',model=model,tokenizer=tokenizer)


style = {'height': 200}
context = st.text_area("Enter Context", height=200)
question = st.text_input("Enter your question")


if(question != ""):
    st.write("Answer of the question: ")
    output = nlp({
        'question': question,
        'context': context
    })

    st.write(output["answer"])
