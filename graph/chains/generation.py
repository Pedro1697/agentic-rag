from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser # this function transform the llm output in a simple and direct str
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()




