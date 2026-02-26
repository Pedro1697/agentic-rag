from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class GradeHallucinations(BaseModel):
    """binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes'or 'no'"
    )

# basically the answer we will get back from the llm chain will format it as the Pydantic class
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of documents 
            Give a binary score 'yes'or 'no'. 'Yes' means that the answer is grounded in / supported by  setof facts"""


hallucination_prompt = ChatPromptTemplate(
    [
        ("system",system),
        ("human","Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ]
)

hallucination_grader : RunnableSequence = hallucination_prompt | structured_llm_grader