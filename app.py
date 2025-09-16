import streamlit as st
from agent_workflow import app
import time

st.set_page_config(page_title="Drug Repurposing Agent", page_icon="🧬", layout="wide")

st.title("🧬 Drug Repurposing AI Agent")
st.markdown("Find new uses for existing drugs using AI-powered analysis")

# Input section
with st.form("input_form"):
    query = st.text_input("Enter a disease or target protein:", 
                         placeholder="e.g., 'Parkinson's disease' or 'EGFR'")
    submitted = st.form_submit_button("Analyze")

# Results section
if submitted and query:
    with st.spinner("🤖 AI agents are working... This may take a minute"):
        try:
            # Run the complete LangGraph workflow
            result = app.invoke({"input": query})
            
            # Display results
            st.success("Analysis complete!")
            
            # Show targets if found
            if result.get('targets'):
                with st.expander("🎯 Identified Targets"):
                    for target in result['targets']:
                        st.write(f"- **{target['gene_symbol']}** (Association score: {target['association_score']})")
            
            # Show compounds if found
            if result.get('compounds'):
                with st.expander("💊 Candidate Compounds"):
                    for compound in result['compounds']:
                        st.write(f"- **{compound['drug_name']}** (Target: {compound['target_gene']}, Mechanism: {compound['mechanism']})")
            
            # Show predictions if available
            if result.get('predictions') and result.get('compounds'):
                with st.expander("🧪 Activity Predictions"):
                    for i, (compound, prediction) in enumerate(zip(result['compounds'], result['predictions'])):
                        st.write(f"- **{compound['drug_name']}**: {prediction*100:.1f}% probability of activity")
            
            # Show the final AI-generated report
            st.divider()
            st.subheader("📊 AI Analysis Report")
            st.markdown(result.get('report', 'No report generated'))
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
