from typing import Any, Dict
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState
from dotenv import load_dotenv

load_dotenv()

web_search_tool = TavilySearch(max_result=3)


def web_search(state:GraphState) -> Dict[str,Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    tavily_response = web_search_tool.invoke({"query":question})
    #the tavily_response is a dict with a list of dicts where the results live, so we need to extract this field to join the content
    tavily_results = tavily_response.get("results",[])
    joined_tavily_result ="\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        #if we not find any relevant document we are going to append our search
        documents = [web_results]

    return {"documents":documents,"question":question}




if __name__ == "__main__":
    web_search(state={"question":"agent memory","documents":None})
