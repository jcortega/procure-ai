import os
from os import listdir
from os.path import isfile, join

from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper


from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from bs4 import BeautifulSoup


from .rfp import Rfp
from utils import bedrock

# TODO: Move to main or config
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
os.environ["AWS_PROFILE"] = "dev"

boto3_bedrock = bedrock.get_bedrock_client(
    region=os.environ.get("AWS_DEFAULT_REGION", None),
)

llm = Bedrock(
    model_id="anthropic.claude-v2",
    client=boto3_bedrock,
    model_kwargs={
        "max_tokens_to_sample": 500,
        "temperature": 0,  # Using 0 to get reproducible results
        "stop_sequences": ["\n\nHuman:"]
    }
)
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=boto3_bedrock)


class Agent:
    current_rfp = None
    rfp_vstore = None

    def __init__(self):
        # self.rfp = rfp
        pass

    def _extract_criteria(self, response: str, tag: str):
        soup = BeautifulSoup(response, "lxml")
        results = soup.find_all(tag)
        if not results:
            return []

        return [{"description": res.find('description').get_text(), "questions": res.find('questions').get_text(), "percentage": int(res.find('percentage').get_text())} for res in results]

    def read_rfp(self, rfp: Rfp):
        """
        Initializes vector store from given documents in rfp
        """

        self.current_rfp = rfp
        print(self.current_rfp)
        loader = PyPDFLoader(f"./rfp/{self.current_rfp.id}")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=1000,
            chunk_overlap=100,
        )
        docs = text_splitter.split_documents(documents)

        # Post load debug log
        # def avg_doc_length(documents): return sum(
        #     [len(doc.page_content) for doc in documents])//len(documents)
        # avg_char_count_pre = avg_doc_length(documents)
        # avg_char_count_post = avg_doc_length(docs)
        # print(
        #     f'Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.')
        # print(
        #     f'After the split we have {len(docs)} documents more than the original {len(documents)}.')
        # print(
        #     f'Average length among {len(docs)} documents (after split) is {avg_char_count_post} characters.')

        # Load document to vector store
        self.rfp_vstore = FAISS.from_documents(
            docs,
            bedrock_embeddings,
        )

    def generate_criteria(self):
        prompt_template = """
        Human: You are a Procurement Specialist Assistant. Use the following pieces of context to provide a concise answer to the question at the end.
        The given document is a Request for Proposal (RFP) document.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        <context>
        {context}
        </context>

        Question: {question}

        Assistant:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.rfp_vstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        # query = """
        # You are a procument specialist, create a set of criteria with percent distribution for choosing the best vendor for the said RFP.
        # The total distribution should be 100.
        # """
        query = """
        Create a criteria with percent distribution for choosing the best vendor for the said RFP.
        Also, for each criterion, provide three guide questions for vendor.
        Total distribution should be 100%. Do not provide range of percentage.
        Return the list in xml format between <criteria></criteria> XML tags,
        with each criterion in <criterion></criterion> XML tags.
        Each criterion should have <description></description>, <questions></questions> and <percentage></percentage> XML tags.
        """

        answer = qa({"query": query})
        # print(answer["result"])

        criteria = self._extract_criteria(answer["result"], 'criterion')

        self.current_rfp.clear_rfp(self.current_rfp.id)
        for c in criteria:
            self.current_rfp.insert_criteria(
                self.current_rfp.id, c["description"], c["questions"], c["percentage"])
        # TODO: validate and repeat until valid

        return criteria

    def evaluate_responses(self, rfp: Rfp, criteria):
        responses_dir = f"responses/{rfp.id}"
        onlyfiles = [f for f in listdir(
            responses_dir) if isfile(join(responses_dir, f))]

        for resp_file_path in onlyfiles:
            complete_path = f"{responses_dir}/{resp_file_path}"
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"Evaluating {complete_path}")
            # resp_file_path = f"responses/{rfp.id}/companyA.txt"
            f = open(complete_path, "r")
            response = f.read()
            f.close()

            prompt_template = """
            Human: You are a Procurement Specialist. Use the following pieces of context and criteria to evaluate vendor response.
            The given document is a Request for Proposal (RFP) document.
            If you don't know how to evaluate, just say that you don't know, don't try to make up an answer.
            <context>
            {context}
            </context>

            Question: {question}

            Assistant:"""

            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=[
                    "context", "question"],
            )
            # PROMPT.partial(response=response, criteria=criteria)

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.rfp_vstore.as_retriever(
                    search_type="similarity", search_kwargs={"k": 3}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )

            query = f"""
            For the given criteria and response below, generate a score for each criteria based on the response,
            and provide the total score in the end.
            <criteria>
            {criteria}
            </criteria>

            <response>
            {response}
            </response>
            """

            answer = qa({"query": query})
            print(answer["result"])
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
