from dotenv import load_dotenv
import os
import io
import requests
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS #facebook AI similarity search
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceEndpoint
# from langchain import HuggingFaceHub

def download_file_from_google_drive(file_id):
    URL = "https://docs.google.com/uc?export=download&confirm=1"
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)
    return response.text

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask Your PDF")

    url = 'https://journalofbigdata.springeropen.com/counter/pdf/10.1186/s40537-022-00683-3.pdf'
    response = requests.get(url)
    pdf = response.content
    # pdf = st.file_uploader("Upload your pdf",type="pdf", key = "pdfuploader")
    #text = download_file_from_google_drive('1FK3_FsiHzICCDRmSU4USsYHYPjgR1xX8JqZ-n3fhIrA')
    
    if pdf is not None and response.ok:
        pdfcontent = io.BytesIO(pdf)
        pdf_reader = PdfReader(pdfcontent)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    
     # if text is not None:
        # spilit ito chuncks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embedding
        embeddings = HuggingFaceEmbeddings()

        knowledge_base = FAISS.from_texts(chunks,embeddings)

        user_question = st.text_input("Ask Question about your PDF:", key="user question")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = HuggingFaceEndpoint(repo_id="google/flan-t5-large", temperature=5,
                                                      max_length=64)
            chain = load_qa_chain(llm,chain_type="stuff")
            response = chain.run(input_documents=docs,question=user_question)

            st.write(response)



        # st.write(chunks)

if __name__ == '__main__':
    main()
