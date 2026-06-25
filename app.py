import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
#from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
# import traceback


st.title("Semantic Search Engine")
st.header("Upload a file to get started.", divider="gray")


#text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

# Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
#embedding_model = OpenAIEmbeddings(model="-embedding-3-large")

# Vector store
chroma_vector_store = Chroma(
    collection_name="my_docs",
    embedding_function=embedding_model,
    persist_directory="./chroma/db"
)

# LLM Model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

uploaded_file = st.file_uploader("Select a file:")


if uploaded_file is not None:
    with st.spinner("Processing file..."):
        try:
            print("File info: ", uploaded_file)

            #save file in memory
            temp_file_path = uploaded_file.name

            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # PDF file loader
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            #print("Docs: ", docs)

            # create chunks
            chunks = text_splitter.split_documents(docs)

            # create embeddings
            # emb1 = embedding_model.embed_query(chunks[0].page_content)
            # print(emb1)

            #index embeddings
            chroma_ids = chroma_vector_store.add_documents(documents=chunks)

            retriever = chroma_vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2}
            )

            if prompt := st.chat_input("Prompt"):
                print(prompt)

                docs_retrieved = retriever.invoke(prompt)

                context = "\n\n".join(
                    [doc.page_content for doc in docs_retrieved]
                )

                final_prompt = f"""
                    You are a helpful assistant.

                    Answer the question using only the context below.

                    Question:
                    {prompt}

                    Context:
                    {context}

                    If the answer is not found in the context, say:
                    "I don't have that information."
                """



                #create a prompt template
                # system_prompt = """You're a helpful assistant. Please answer the following question {question}, only using the following information {document}.
                # If you cant answer the question, just say you don't have that information."""

                # prompt_template = ChatPromptTemplate.from_messages(
                #     [
                #         ("system", system_prompt)
                #     ]
                # )

                # print("Prompt:", prompt)
                # print("Retrieved Docs:", docs_retrieved)

                # final_prompt = prompt_template.invoke({
                #     "question": prompt,
                #     "document": docs_retrieved
                #     # "context": context
                     
                # })

                # UI container
                result_placeholder = st.empty()


                #create complition
                # complition = llm.invoke(final_prompt)
                # print("Completion", complition)

                # streaming the complition result
                full_complition = ""
                print(type(final_prompt))
                print(final_prompt)
                for chunk in llm.stream(final_prompt):
                    full_complition += chunk.content
                    result_placeholder.write(full_complition)


        except Exception as e:
            print(e)
            # traceback.print_exc()

        # Removes file after it is done

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)