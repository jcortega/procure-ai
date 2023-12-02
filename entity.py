import os
import sys

from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww


os.environ["AWS_DEFAULT_REGION"] = "us-west-2"  
os.environ["AWS_PROFILE"] = "dev"
# os.environ["BEDROCK_ASSUME_ROLE"] = "<YOUR_ROLE_ARN>"  # E.g. "arn:aws:..."


boto3_bedrock = bedrock.get_bedrock_client(
    region=os.environ.get("AWS_DEFAULT_REGION", None),
)

from langchain.llms.bedrock import Bedrock

# - create the Anthropic Model
llm = Bedrock(
    model_id="anthropic.claude-v2",
    client=boto3_bedrock,
    model_kwargs={
        "max_tokens_to_sample": 200,
        "temperature": 0, # Using 0 to get reproducible results
        "stop_sequences": ["\n\nHuman:"]
    }
)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=boto3_bedrock)




loader = PyPDFDirectoryLoader("./rfp/")

documents = loader.load()
# - in our testing Character split works better with this PDF data set
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 100,
)
docs = text_splitter.split_documents(documents)


avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents])//len(documents)
avg_char_count_pre = avg_doc_length(documents)
avg_char_count_post = avg_doc_length(docs)
print(f'Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.')
print(f'After the split we have {len(docs)} documents more than the original {len(documents)}.')
print(f'Average length among {len(docs)} documents (after split) is {avg_char_count_post} characters.')


sample_embedding = np.array(bedrock_embeddings.embed_query(docs[0].page_content))
print("Sample embedding of a document chunk: ", sample_embedding)
print("Size of the embedding: ", sample_embedding.shape)


from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

vectorstore_faiss = FAISS.from_documents(
    docs,
    bedrock_embeddings,
)

wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)


from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

prompt_template = """

Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

query = """The given document is a Request for Proposal (RFP) document. You are a Procumente Specialist, create a criteria with percent distribution for choosing the best vendor for the said RFP"""

answer = qa({"query": query})
print(answer["result"])

# #########
# query = f"""

# Human: Given the email inside triple-backticks, please read it and analyse the contents.
# If a name of a book is mentioned, return it, otherwise return nothing.

# Email: ```
# {book_question_email}
# ```

# Assistant:"""
# result = llm(query)
# print(result.strip())
# #######