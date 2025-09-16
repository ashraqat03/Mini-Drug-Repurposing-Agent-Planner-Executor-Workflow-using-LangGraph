from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from tools import find_targets, find_compounds, predict_activity
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# Define State
class AgentState(TypedDict):
    input: str
    next_step: str
    targets: List[dict]
    compounds: List[dict]
    predictions: List[dict]
    report: str

# Initialize Graph
workflow = StateGraph(AgentState)

# Define Nodes
def planner_node(state):
    planner_prompt = """You are an expert planner for a drug discovery AI. Analyze the user's input and decide the first step. Rules:
    1. If input is a disease (e.g., "Parkinson's", "cancer"), output: 'find_targets'
    2. If input is a protein/gene/target (e.g., "EGFR", "BRCA1"), output: 'find_compounds'
    3. If unsure, output 'find_targets'
    Output only one word: 'find_targets' or 'find_compounds'."""
    
    response = llm.invoke([{"role": "user", "parts": [planner_prompt]},
                          {"role": "user", "parts": [f"User input: {state['input']}"]}])
    return {"next_step": response.content.strip().lower()}

def find_targets_node(state):
    result = find_targets(state['input'])
    return {"targets": result["targets"]} if "targets" in result else {"error": result["error"]}

def find_compounds_node(state):
    target_genes = [t['gene_symbol'] for t in state['targets']] if state.get('targets') else [state['input']]
    result = find_compounds(target_genes)
    return {"compounds": result["compounds"]} if "compounds" in result else {"error": result["error"]}

def predict_activity_node(state):
    smiles_list = [c['SMILES'] for c in state['compounds']]
    result = predict_activity(smiles_list)
    return {"predictions": result["predictions"]} if "predictions" in result else {"error": result["error"]}

def reporter_node(state):
    context = f"""
    USER QUERY: {state['input']}
    TARGETS: {state.get('targets', [])}
    COMPOUNDS: {state.get('compounds', [])}
    PREDICTIONS: {state.get('predictions', [])}
    """
    
    prompt = """You are an expert biomedical research assistant. Create a concise drug repurposing report with:
    1. Executive summary
    2. Hypothesis behind proposed targets
    3. Table of top candidates with targets and scores
    4. 2-3 next steps for validation
    Base your report ONLY on the data provided."""
    
    try:
        response = llm.invoke(prompt + context)
        return {"report": response.content}
    except Exception as e:
        return {"error": f"Failed to generate report: {str(e)}"}

# Add nodes to graph
workflow.add_node("planner", planner_node)
workflow.add_node("find_targets", find_targets_node)
workflow.add_node("find_compounds", find_compounds_node)
workflow.add_node("predict_activity", predict_activity_node)
workflow.add_node("reporter", reporter_node)

# Set up graph flow
workflow.set_entry_point("planner")

def decide_next_step(state):
    return state['next_step']

workflow.add_conditional_edges(
    "planner",
    decide_next_step,
    {"find_targets": "find_targets", "find_compounds": "find_compounds"}
)

workflow.add_edge("find_targets", "find_compounds")
workflow.add_edge("find_compounds", "predict_activity")
workflow.add_edge("predict_activity", "reporter")
workflow.add_edge("reporter", END)

# Compile the workflow
app = workflow.compile()
