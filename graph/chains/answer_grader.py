from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="Answer addresses the question 'yes'or 'no'"
    )

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a quesation \n
            Give a binary score 'yes'or 'no'. 'Yes'means that the answer resolves the quesation."""

answer_prompt = ChatPromptTemplate(
    [
        ("system",system),
        ("human","User question: \n\n {question} \n\n LLM generation: {generation}")
    ]
)

answer_grader : RunnableSequence = answer_prompt | structured_llm_grader