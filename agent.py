import json
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Define State Structure
class AgentState(TypedDict):
    selected_station: str
    time_details: str
    predicted_energy: float
    station_avg_load: float
    is_high_load: bool
    retrieved_guidelines: str
    final_output: dict
    groq_api_key: str

# Node 1: Analyze load
def analyze_node(state: AgentState):
    # Rule based logic to identify high load
    # Let's say if predicted energy is 10% above average, it's high load
    is_high = state['predicted_energy'] > (state['station_avg_load'] * 1.05)
    return {"is_high_load": is_high}

# Node 2: Retrieve Guidelines
def retrieve_node(state: AgentState):
    try:
        # Lightweight FastEmbed embeddings
        embeddings = FastEmbedEmbeddings()
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # Query based on the load
        if state['is_high_load']:
            query = f"guidelines for high-demand mitigation and expansion at {state['time_details']}"
        else:
            query = f"guidelines for scheduling and operations optimization at {state['time_details']}"
            
        docs = retriever.invoke(query)
        retrieved_texts = "\n---\n".join([doc.page_content for doc in docs])
        return {"retrieved_guidelines": retrieved_texts}
    except Exception as e:
        return {"retrieved_guidelines": f"Could not retrieve guidelines: {str(e)}"}

# Node 3: LLM Reason and Generate Plan
def generation_node(state: AgentState):
    try:
        if not state.get('groq_api_key'):
            return {"final_output": {"error": "Groq API Key is not set."}}
            
        llm = ChatGroq(
            api_key=state['groq_api_key'],
            model="llama-3.3-70b-versatile",
            temperature=0.2
        )
        
        system_prompt = """You are an AI EV Infrastructure Planning Assistant. 
You will receive charging demand data for a station, its historical average, and institutional infrastructure planning guidelines.
Your goal is to reason about this data and output a structured infrastructure expansion or scheduling optimization plan.

You MUST structure your response as valid JSON with EXACTLY the following keys:
- "Analysis": A brief summary of the charging demand context.
- "Locate": The ID or name of the high-load location.
- "Plan": Your infrastructure expansion recommendations based on the guidelines.
- "Optimize": Scheduling and operational insights based on the guidelines.
- "Refs": Supporting references (cite the provided guidelines).

Only return RAW JSON, with no markdown formatting around it."""

        user_prompt = f"""
Station Name: {state['selected_station']}
Time Context: {state['time_details']}
Predicted Energy Demand: {state['predicted_energy']:.2f} kWh
Historical Average Load: {state['station_avg_load']:.2f} kWh
Identified as High-Load Scenario: {state['is_high_load']}

=== Retrieved Guidelines ===
{state['retrieved_guidelines']}
============================

Based on the data and guidelines, generate the JSON output.
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        
        chain = prompt | llm
        response = chain.invoke({})
        
        try:
            # We enforce JSON output, sometimes the LLM puts it in markdown
            content = response.content
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            parsed_json = json.loads(content.strip())
            return {"final_output": parsed_json}
        except Exception as parse_error:
            # Fallback if json parsing fails
            return {"final_output": {
                "error": "Failed to parse LLM output into JSON.",
                "raw_response": response.content
            }}
    except Exception as e:
        return {"final_output": {"error": str(e)}}

# Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("analyze", analyze_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generation_node)

workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
agent_chain = workflow.compile()

def run_agentic_workflow(station: str, time_context: str, predicted: float, avg: float, groq_key: str):
    initial_state = {
        "selected_station": station,
        "time_details": time_context,
        "predicted_energy": predicted,
        "station_avg_load": avg,
        "is_high_load": False,
        "retrieved_guidelines": "",
        "final_output": {},
        "groq_api_key": groq_key
    }
    
    final_state = agent_chain.invoke(initial_state)
    return final_state['final_output']
