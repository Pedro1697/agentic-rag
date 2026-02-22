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

