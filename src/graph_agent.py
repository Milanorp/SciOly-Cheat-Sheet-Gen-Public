import os
import sys
import signal 
import arxiv

# Add the project root to sys.path to allow 'from src.X import Y' imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma 
from langchain_core.tools import tool, ToolException
from langchain_community.tools import DuckDuckGoSearchRun
from src.factory import factory

# 0. LOAD SECRETS & SETUP
load_dotenv()
config = factory.get_config()

# --- DISABLE KEYBOARD INTERRUPT (Ctrl+C) ---
signal.signal(signal.SIGINT, signal.SIG_IGN)
# -------------------------------------------

print(f"Waking up {config['models']['researcher']}...")
llm = factory.get_llm(purpose="researcher")

# =====================================================================
# 1. THE DATABASE & FAST RETRIEVAL PIPELINE
# =====================================================================
embeddings = factory.get_embeddings()
vectorstore = Chroma(persist_directory=config['database']['db_path'], embedding_function=embeddings)

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

# Headroom Context Compression Helper
def compress_if_enabled(text: str, strategy: str = "general") -> str:
    use_compression = config.get("research", {}).get("enable_context_compression", False)
    if not use_compression:
        return text
    try:
        from headroom import compress_content
        if not text.strip():
            return text
        before_len = len(text)
        compressed = compress_content(text, strategy=strategy)
        after_len = len(compressed)
        print(f"      [Headroom] Compressed: {before_len} -> {after_len} chars ({round((1 - after_len/before_len)*100, 1)}% saved)")
        return compressed
    except Exception:
        return text

# Intelligent LLM Tool Summarizer
def summarize_tool_output(query: str, raw_text: str) -> str:
    """Summarizes tool output using a cheap fast LLM to prevent context explosion."""
    if not config.get("research", {}).get("enable_llm_tool_summarization", False):
        return compress_if_enabled(raw_text, strategy="rag")
        
    if not raw_text or len(raw_text) < 500:
        return compress_if_enabled(raw_text, strategy="rag")
        
    print(f"      [LLM Summarizer] Compressing massive context for: '{query}'...")
    try:
        summarizer_llm = factory.get_llm(purpose="researcher")
        summarizer_llm.temperature = 0.0
        
        prompt = SystemMessage(content=f"""You are a data compressor. Extract ONLY the facts, numbers, and formulas highly relevant to the query: '{query}'.
        Rules:
        - Output a raw, dense bulleted list.
        - Ignore fluff, introductions, or unrelated text.
        - Keep it strictly under 150 words.
        - DO NOT lose any technical constants, numbers, or specific chemical/physics formulas.""")
        
        res = summarizer_llm.invoke([prompt, HumanMessage(content=raw_text)])
        compressed = res.content
        print(f"      [LLM Summarizer] Reduced {len(raw_text)} chars to {len(compressed)} chars.")
        return compressed
    except Exception as e:
        print(f"      [LLM Summarizer Error] {e}")
        return compress_if_enabled(raw_text, strategy="rag")

@tool
def search_scioly_rules(search_query: str, event_metadata: str = None) -> str:
    """Searches the official Science Olympiad rulebook."""
    print(f"\n[🔧 TOOL] Searching rules for: '{search_query}'")
    try:
        # Ensure path is absolute
        db_path = os.path.abspath(config['database']['db_path'])
        temp_vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

        # Direct search (avoiding query expansion LLM calls to reduce cost/tokens)
        docs = []
        if event_metadata:
            try:
                docs = temp_vectorstore.similarity_search(search_query, k=3, filter={"Event": event_metadata.title()})
            except:
                docs = []
        
        if not docs:
            docs = temp_vectorstore.similarity_search(search_query, k=3)
            
        if not docs:
            return "No relevant rulebook sections found matching that query."

        raw_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
        return summarize_tool_output(search_query, raw_text)
        
    except Exception as e:
        return f"Error during search: {e}"

@tool
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
        raw_text = "\n---\n".join(results)
        return summarize_tool_output(query, raw_text)
    except Exception as e:
        return f"Error searching ArXiv: {e}"

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
    raw = ddg.run(f"{query} site:scioly.org/wiki")
    return summarize_tool_output(query, raw)

@tool
def search_academic_biology(query: str) -> str:
    """Use ONLY as a backup if ArXiv fails."""
    print(f"\n[🔬 SNIPER TOOL] Searching Medical/Bio Sites for: '{query}'")
    raw = ddg.run(f"{query} site:ncbi.nlm.nih.gov OR site:.edu")
    return summarize_tool_output(query, raw)

@tool
def search_physics_and_engineering(query: str) -> str:
    """Use ONLY as a backup if ArXiv fails."""
    print(f"\n[SNIPER TOOL] Searching Engineering Sites for: '{query}'")
    raw = ddg.run(f"{query} 'physics' OR 'engineering' site:.edu OR site:.gov")
    return summarize_tool_output(query, raw)

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

@tool
def search_past_tests(search_query: str, event_metadata: str = None) -> str:
    """
    Searches through a database of actual past Science Olympiad tests. 
    Use this to see EXACTLY how a topic has been questioned in the past, what constants were provided, and what level of detail was required.
    """
    print(f"\n[🔬 TOOL] Searching Past Tests for: '{search_query}'")
    try:
        db_path = os.path.abspath(config['database']['db_path'])
        temp_vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        # Filter for documents where Source_Type is "Past Test"
        search_filter = {"Source_Type": "Past Test"}
        if event_metadata:
             search_filter = {
                 "$and": [
                     {"Source_Type": "Past Test"},
                     {"Event": event_metadata.title()}
                 ]
             }

        docs = temp_vectorstore.similarity_search(search_query, k=2, filter=search_filter)
        
        if not docs:
            return "No specific examples found in past tests for this query."

        results = []
        for doc in docs:
            filename = doc.metadata.get("Filename", "Unknown Test")
            results.append(f"--- From Test: {filename} ---\n{doc.page_content}\n")
            
        raw_text = "\n\n".join(results)
        return summarize_tool_output(search_query, raw_text)
    except Exception as e:
        return f"Error searching past tests: {e}"

tools = [
    reject_out_of_scope,
    search_scioly_rules,
    search_past_tests,
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

def build_app(cache_info=None):
    # Retrieve LLM (potentially with cache)
    graph_llm = factory.get_llm(purpose="researcher", cache_info=cache_info)
    llm_with_tools = graph_llm.bind_tools(tools)

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

    return workflow.compile()

# Global un-cached fallback for legacy imports
app = build_app()

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
