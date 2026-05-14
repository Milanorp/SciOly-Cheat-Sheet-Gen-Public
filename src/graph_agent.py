import os
import signal 
import arxiv # <-- ADDED THIS IMPORT
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma 
from langchain_core.tools import tool, ToolException
from langchain_community.tools import DuckDuckGoSearchRun

# 0. LOAD SECRETS & SETUP
load_dotenv()

# --- DISABLE KEYBOARD INTERRUPT (Ctrl+C) ---
signal.signal(signal.SIGINT, signal.SIG_IGN)
# -------------------------------------------

print("Waking up Gemini 2.5 Flash...")
# OPTIMIZATION 1: Added max_retries for graceful rate-limit handling
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=3)

# =====================================================================
# 1. THE DATABASE & FAST RETRIEVAL PIPELINE
# =====================================================================
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(persist_directory="./scioly_db", embedding_function=embeddings)

# Base Search Engine
ddg = DuckDuckGoSearchRun()

# =====================================================================
# 2. THE HIDDEN CLAUDE-STYLE TOOLKIT
# =====================================================================

@tool
def reject_out_of_scope() -> str:
    """Use this tool IMMEDIATELY if the user asks a question NOT related to Science Olympiad."""
    print(f"\n[🛑 TOOL] Scope Violation Detected. Triggering Kill Switch.")
    return "SYSTEM ERROR: This request is outside my operational scope. Tell the user you can only assist with Science Olympiad research."

@tool(handle_tool_error=True)
def search_scioly_rules(search_query: str, event_metadata: str = None) -> str:
    """Searches the official Science Olympiad rulebook."""
    print(f"\n[🔧 TOOL] Multi-Query Expanding: '{search_query}'")
    try:
        # Step 1: Query Expansion
        expansion_prompt = f"Generate 3 highly diverse search queries to find rules related to '{search_query}' for the Science Olympiad event: {event_metadata}. Focus on technical specifications, penalties, and allowed materials. Return ONLY the queries, one per line."
        expansion_res = llm.invoke(expansion_prompt)
        queries = [q.strip().strip('- ') for q in expansion_res.content.split('\n') if q.strip()]
        queries.append(search_query) # Always include original
        
        # Deduplicate queries
        queries = list(set(queries))
        
        all_docs = []
        search_kwargs = {"k": 4} 
        if event_metadata:
            search_kwargs["filter"] = {"Event": event_metadata.title()}
            print(f"        > Multi-Querying with Metadata: {event_metadata.title()}")
            
        for q in queries:
            print(f"        > Sub-query: {q[:50]}...")
            docs = vectorstore.similarity_search(q, **search_kwargs)
            all_docs.extend(docs)
        
        # Step 2: Content Deduplication
        unique_contents = {}
        for doc in all_docs:
            if doc.page_content not in unique_contents:
                unique_contents[doc.page_content] = doc

        if not unique_contents:
            return "No relevant rulebook sections found matching those queries."

        return "\n\n---\n\n".join([doc.page_content for doc in unique_contents.values()])
        
    except Exception as e:
        raise ToolException(f"Error during multi-query search: {e}")

@tool(handle_tool_error=True)
def search_arxiv(query: str) -> str:
    """
    Searches the live ArXiv database for advanced academic research papers. Use this FIRST for deep physics, biology, and engineering concepts.
    
    CRITICAL SEARCH RULES FOR ARXIV:
    - ArXiv uses strict keyword matching, NOT semantic search.
    - Your query MUST be extremely loose and short (2 to 4 keywords max).
    - NEVER use full sentences, questions, or highly specific constraints.
    - GOOD QUERY: "low Reynolds propeller" OR "micro air vehicle aerodynamics"
    - BAD QUERY: "How does a low Reynolds number environment affect the aerodynamic efficiency"
    """
    print(f"\n[🎓 TOOL] Searching live ArXiv for: '{query}'")
    try:
        search = arxiv.Search(
            query=query,
            max_results=3, # Pull top 3 papers to save context space
            sort_by=arxiv.SortCriterion.Relevance
        )
        client = arxiv.Client()
        results = []
        for paper in client.results(search):
            results.append(f"Title: {paper.title}\nAuthors: {', '.join([a.name for a in paper.authors])}\nAbstract: {paper.summary}\n")
            
        if not results: return "No ArXiv papers found for that query. Try a shorter, broader keyword."
        return "\n---\n".join(results)
    except Exception as e:
        raise ToolException(f"Error searching ArXiv: {e}")

@tool
def request_search_clearance(scientific_domain: str) -> str:
    """You MUST use this tool before conducting ANY external research."""
    print(f"\n[🛡GATEKEEPER] AI requesting clearance for: '{scientific_domain}'")
    domain = scientific_domain.lower()
    if "bio" in domain or "anatomy" in domain or "disease" in domain:
        return "CLEARANCE GRANTED: You should prioritize 'search_arxiv' for advanced biology, using 'search_academic_biology' ONLY as a backup."
    elif "phys" in domain or "build" in domain or "engineer" in domain or "machine" in domain:
        return "CLEARANCE GRANTED: You should prioritize 'search_arxiv' for advanced physics/engineering, using 'search_physics_and_engineering' ONLY as a backup."
    elif "wiki" in domain or "strategy" in domain or "event" in domain:
        return "CLEARANCE GRANTED: You may ONLY use the 'search_scioly_wiki' tool."
    else:
        return "CLEARANCE DENIED: Domain not recognized. Do not search the web. Ask user to clarify."

@tool
def search_scioly_wiki(query: str) -> str:
    """Use this tool FIRST for historical build parameters, event strategies, and past tests."""
    print(f"\n[📚 SNIPER TOOL] Searching SciOly Wiki for: '{query}'")
    return ddg.run(f"{query} site:scioly.org/wiki")

@tool
def search_academic_biology(query: str) -> str:
    """Use ONLY as a backup if ArXiv fails."""
    print(f"\n[🔬 SNIPER TOOL] Searching Medical/Bio Sites for: '{query}'")
    return ddg.run(f"{query} site:ncbi.nlm.nih.gov OR site:.edu")

@tool
def search_physics_and_engineering(query: str) -> str:
    """Use ONLY as a backup if ArXiv fails."""
    print(f"\n[SNIPER TOOL] Searching Engineering Sites for: '{query}'")
    return ddg.run(f"{query} 'physics' OR 'engineering' site:.edu OR site:.gov")

@tool
def submit_final_answer(draft_answer: str, academic_level: str) -> str:
    """You MUST use this tool to evaluate your final answer BEFORE talking to the user."""
    print(f"\n[⚖SELF-GRADER] AI submitted draft at level: '{academic_level}'")
    
    if academic_level == "Graduate Research":
        print("[🛑 REJECTED] Draft is too complex. Forcing rewrite.")
        return "SYSTEM ERROR: Answer rejected. Too complex. Rewrite at a 'High School AP' level and use this tool again."
    if academic_level in ["Elementary", "Middle School"]:
        print("[🛑 REJECTED] Draft is too basic. Forcing rewrite.")
        return "SYSTEM ERROR: Answer rejected. Too basic. Add more technical depth and submit again at 'High School AP' level."
    
    print("[✅ APPROVED] Answer scope is perfectly calibrated for SciOly.")
    return "APPROVED_FINAL_ANSWER: The answer is approved. You may now output this exact draft directly to the user."

tools = [
    reject_out_of_scope,
    search_scioly_rules,
    search_arxiv,
    request_search_clearance,
    search_scioly_wiki,
    search_academic_biology,
    search_physics_and_engineering,
    submit_final_answer
]

# =====================================================================
# 3. THE AI BRAIN & GRAPH SETUP
# =====================================================================
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def ai_node(state: AgentState):
    print("\n[🧠 AI NODE] AI is analyzing tools and thinking...")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools) 

print("Building the Graph Blueprint...")
workflow = StateGraph(AgentState)
workflow.add_node("agent", ai_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent") 

app = workflow.compile()

# =====================================================================
# 4. RUN THE DYNAMIC AGENT (MUTED FOR IMPORT)
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADVANCED SCIENCE OLYMPIAD AI TERMINAL")
    print("="*60)

    # 1. Ask for the Event First
    current_event = input("\nWhat Science Olympiad event are you studying for? ")
    
    # 2. Ask the Question Second
    user_question = input("❓ What is your question? ")

    # 3. Define the System Prompt using the Event Name
    system_prompt = SystemMessage(content=f"""You are an expert Science Olympiad AI Assistant. 
    The user is studying for the event: {current_event}.

    STRICT WORKFLOW PROTOCOL:
    1. SCOPE CHECK: If out of scope, call 'reject_out_of_scope'.
    2. CHECK RULES: Always use 'search_scioly_rules' first. You MUST pass "{current_event}" into the event_metadata parameter.
       - CRITICAL SEARCH RULE: DO NOT put the entire user question into the search tool. Break complex questions down.
    3. REQUEST CLEARANCE: If you need web research, use 'request_search_clearance'.
    4. EXTERNAL RESEARCH PRIORITY: 
       - ALWAYS prioritize using 'search_arxiv' for advanced theory.
       - ONLY use the other web sniper tools if ArXiv returns no useful results.
       - BE EFFICIENT: Do not spam search tools. Execute ONE highly targeted search at a time.
    5. GATEKEEPER CHECK: Use 'submit_final_answer' to self-grade.
    6. FINAL OUTPUT: Directly output the approved text to the user.""")

    # 4. Initialize State
    initial_state = {"messages": [system_prompt, HumanMessage(content=user_question)]}

    # 5. Invoke the AI
    final_state = app.invoke(initial_state, config={"recursion_limit": 15})

    print("\n================ FINAL ANSWER ================\n")
    print(final_state["messages"][-1].content)
    print("\n==============================================")