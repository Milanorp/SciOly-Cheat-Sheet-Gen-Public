from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

# 1. Connect to the database
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="./scioly_db", embedding_function=embeddings)

# 2. Define the tool (Renamed the argument to 'search_query' so the AI understands it better)
@tool
def search_scioly_rules(search_query: str) -> str:
    """
    Searches the official Science Olympiad rulebook. 
    INSTRUCTIONS: You must pass a specific, plain text string. Do NOT pass dictionaries.
    Example: "Chemistry event allowed materials and equipment penalty"
    """
    print(f"\n[🔧 TOOL TRIGGERED] Searching database for: '{search_query}'")
    docs = vectorstore.similarity_search(search_query, k=5)
    return "\n\n".join([doc.page_content for doc in docs])

# 3. Wake up the Agent
print("Waking up Llama 3.1...")
llm = ChatOllama(model="llama3.1", temperature=0) # Temperature 0 makes it more logical and less prone to typos
agent = llm.bind_tools([search_scioly_rules])

# 4. Create the "Memory" with a strict SYSTEM PROMPT
user_question = "What materials am I allowed to bring to the Chemistry event?"
print(f"\n[USER]: {user_question}")

messages = [
    SystemMessage(content="You are a strict Science Olympiad AI. If you need to search the rules, you MUST use the search_scioly_rules tool. Always pass a valid text string for your search query."),
    HumanMessage(content=user_question)
]

# 5. Turn 1: AI decides what to do
print("Waiting for AI to think...")
ai_msg = agent.invoke(messages)
messages.append(ai_msg)

# 6. The Agent Loop:
if ai_msg.tool_calls:
    for tool_call in ai_msg.tool_calls:
        
        # --- THE SAFETY NET ---
        try:
            # We attempt to run the tool
            args = tool_call["args"]
            
            # If the AI hallucinates a dictionary again, we catch it manually
            if isinstance(args.get("search_query"), dict):
                print("⚠️ [WARNING] The AI hallucinated the JSON schema. Forcing it to use a fallback query...")
                args["search_query"] = "Chemistry event equipment allowed materials"
                
            tool_output = search_scioly_rules.invoke(args)
            messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
            
        except Exception as e:
            # If it completely crashes, we don't kill the program. We feed the error back to the AI!
            print(f"⚠️ [ERROR CATCHED]: {e}")
            messages.append(ToolMessage(content="Error: You formatted your tool input incorrectly. Please try again with a plain string.", tool_call_id=tool_call["id"]))
        # ----------------------
        
    # 7. Turn 2: Give the text back to the AI for the final answer
    print("\n[🧠] Feeding database results back to the AI for synthesis...")
    final_response = agent.invoke(messages)
    
    print("\n================ FINAL ANSWER ================\n")
    print(final_response.content)
    print("\n==============================================")

else:
    print("\n[AI]:", ai_msg.content)