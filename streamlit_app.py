import os
import glob
import json
import streamlit as st
import pymupdf
import docx
import pandas as pd
from neo4j import GraphDatabase
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+ssc://51204372.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "51204372")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="GraphAI | Enterprise Intelligence Platform",
    page_icon="🕸️",  # <--- Change favicon here
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title { font-size: 2.6rem !important; font-weight: 800; color: #FFFFFF; margin-bottom: 0px; }
    .sub-title { font-size: 1.2rem !important; font-weight: 500; color: #4285F4; margin-bottom: 25px; }
    
    .stButton>button {
        width: 100%; border-radius: 8px; height: 3.2em; background-color: #1A73E8; color: white; font-weight: 700; font-size: 1.05rem; border: none; transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: #1557B0; box-shadow: 0px 4px 12px rgba(26, 115, 232, 0.4); }
    
    .metric-card {
        background: linear-gradient(135deg, #1E1E2E 0%, #11111B 100%); padding: 18px 20px; border-radius: 12px; border-left: 4px solid #4285F4; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-title { font-size: 0.85rem; color: #A0A0A0; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 1.35rem; color: #FFFFFF; font-weight: 700; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# UNIVERSAL TEMPORARY FILE CLEANUP HELPER
# ==========================================
def cleanup_temp_files():
    """Finds and automatically deletes all temporary upload files regardless of extension."""
    for temp_file in glob.glob("temp_*"):
        try:
            os.remove(temp_file)
        except Exception:
            pass

@st.cache_resource
def get_neo4j_driver():
    return GraphDatabase.driver(
        NEO4J_URI, 
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        max_connection_lifetime=30,
        liveness_check_timeout=10
    )

driver = get_neo4j_driver()

def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# FILE PARSER HELPER (PDF, DOCX, CSV, TXT, MD)
# ==========================================
def extract_text_from_file(uploaded_file):
    filename = uploaded_file.name.lower()
    
    if filename.endswith(".pdf"):
        with open("temp_ingest.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        doc = pymupdf.open("temp_ingest.pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text.strip(), "pdf"
        
    elif filename.endswith(".docx"):
        with open("temp_ingest.docx", "wb") as f:
            f.write(uploaded_file.getbuffer())
        doc = docx.Document("temp_ingest.docx")
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return text.strip(), "docx"
        
    elif filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        text = df.to_string(index=False)
        return text.strip(), "csv"
        
    elif filename.endswith(".txt") or filename.endswith(".md"):
        text = uploaded_file.read().decode("utf-8")
        return text.strip(), "txt"
        
    return "", "unknown"

# ==========================================
# UNIVERSAL EXTRACTION & NEO4J INGESTION
# ==========================================
def extract_universal_knowledge_graph(text_chunk):
    client = get_gemini_client()
    prompt = f"""
    Analyze the following document text and extract structured knowledge graph entities and relationships.
    This could be a trade document, bank statement, contract, CSV data, invoice, or official report.

    DOCUMENT TEXT:
    {text_chunk}

    Extract all key entities, facts, and relationships into JSON format with the following structure:
    {{
      "nodes": [
        {{"id": "EntityNameOrValue", "label": "EntityType"}} 
      ],
      "relationships": [
        {{"source": "EntityA", "type": "RELATIONSHIP_TYPE", "target": "EntityB"}}
      ]
    }}

    Return ONLY a valid JSON object. Do not include markdown code block syntax.
    """
    chat = client.chats.create(model='gemini-3.6-flash')
    response = chat.send_message(prompt)
    clean_json = response.text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean_json)

def save_universal_triples_to_neo4j(graph_data):
    with driver.session() as session:
        for node in graph_data.get("nodes", []):
            label = node.get("label", "Entity").replace(" ", "_")
            node_id = node.get("id")
            if not node_id: continue
            session.run(f"MERGE (n:`{label}` {{name: $name}})", name=str(node_id))
                
        for rel in graph_data.get("relationships", []):
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type", "CONNECTED_TO").replace(" ", "_").upper()
            if source and target:
                cypher = f"""
                MATCH (a) WHERE a.name = $source
                MATCH (b) WHERE b.name = $target
                MERGE (a)-[r:`{rel_type}`]->(b)
                """
                session.run(cypher, source=str(source), target=str(target))

# ==========================================
# DYNAMIC GRAPHRAG SEARCH WITH INPUT VALIDATION
# ==========================================
def execute_smart_graphrag(question):
    client = get_gemini_client()
    keywords = [word.strip().lower() for word in question.split() if len(word) > 2]
    
    cypher_retrieval = """
    MATCH (a)-[r]->(b)
    OPTIONAL MATCH (b)-[r2]->(c)
    WITH a, r, b, r2, c,
         [term IN $keywords WHERE toLower(a.name) CONTAINS term OR toLower(b.name) CONTAINS term OR toLower(labels(a)[0]) CONTAINS term OR toLower(labels(b)[0]) CONTAINS term] AS matches
    ORDER BY size(matches) DESC
    RETURN labels(a)[0] AS SourceType, a.name AS Source, 
           type(r) AS Rel1, 
           labels(b)[0] AS TargetType, b.name AS Target,
           CASE WHEN r2 IS NOT NULL THEN type(r2) ELSE null END AS Rel2,
           CASE WHEN c IS NOT NULL THEN labels(c)[0] ELSE null END AS SubTargetType,
           c.name AS SubTarget
    LIMIT 40
    """
    
    graph_facts = []
    with driver.session() as session:
        result = session.run(cypher_retrieval, keywords=keywords)
        for record in result:
            fact = f"[{record['SourceType']}] '{record['Source']}' --({record['Rel1']})--> [{record['TargetType']}] '{record['Target']}'"
            if record['SubTarget']:
                fact += f" --({record['Rel2']})--> [{record['SubTargetType']}] '{record['SubTarget']}'"
            graph_facts.append(fact)
            
    context_str = "\n".join(graph_facts)
    
    prompt = f"""
    You are an Enterprise GraphAI Intelligence System.
    Answer the user's question accurately using ONLY the structured Graph Database facts provided below.

    --- RETRIEVED NEO4J KNOWLEDGE GRAPH FACTS ---
    {context_str}
    --------------------------------------------

    User Query: {question}

    Instructions:
    - Directly and accurately answer the user's query based on the facts.
    - Format your response as a professional, structured analytical report with clear headings.
    - If the context does not contain the requested information, explicitly state what is missing.
    """
    
    chat = client.chats.create(model='gemini-3.6-flash')
    response = chat.send_message(prompt)
    return response.text, graph_facts

# SIDEBAR ARCHITECTURE
st.sidebar.markdown("## ⚙️ Platform Engine Specs")
st.sidebar.markdown("""
- **Knowledge Engine:** Neo4j Aura Cloud GDS
- **Reasoning Model:** Google Gemini 3.6 Flash
- **Architecture:** Universal GraphRAG
- **Multi-Format Parser:** PDF, DOCX, CSV, TXT, MD
- **Auto-Privacy:** Automatic Disk Cleanup
""")

st.sidebar.markdown("---")
st.sidebar.markdown("[🐙 View GitHub Source Code](https://github.com/sonnyhoney/AfCFTA-GraphAI-Lab)")
st.sidebar.markdown("[💼 Connect on LinkedIn](https://www.linkedin.com/in/agwu-eze)")

# MAIN PLATFORM HEADER
st.markdown('<p class="main-title">🕸️ Universal GraphAI Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Enterprise Multi-Format Knowledge Graph & Intelligence System powered by Neo4j & Google Gemini</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-title">Knowledge Engine</div><div class="metric-value">Neo4j GDS Cloud</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-title">Supported Formats</div><div class="metric-value">PDF, DOCX, CSV, TXT</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-title">Export Capabilities</div><div class="metric-value">MD, JSON, TXT</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-title">Inference Engine</div><div class="metric-value">Gemini 3.6 Flash</div></div>', unsafe_allow_html=True)

st.write("")
st.write("")

# PLATFORM TABS
tab1, tab2, tab3 = st.tabs(["🔍 Intelligence Query Engine", "📄 Ingest Multi-Format Document", "📊 Knowledge Graph Inspector"])

# TAB 1: QUERY ENGINE & EXPORT WITH INPUT VALIDATION
with tab1:
    st.markdown("### 💬 Enterprise Query Engine")
    
    sample_scenario = st.selectbox(
        "Select a pre-configured query or type a custom question below:",
        [
            "Custom Search Query...",
            "What are the rules, required documents, and regulatory approvals for exporting Processed Cocoa Powder from Ghana to Nigeria under AfCFTA?",
            "What are the duty rates and documents required for exporting Automotive Parts from South Africa to Kenya?",
            "How can Egypt export Phosphate Fertilizers to Nigeria under AfCFTA?"
        ]
    )

    default_query = ""
    if sample_scenario != "Custom Search Query...":
        default_query = sample_scenario

    user_query = st.text_area("Enter Your Query:", value=default_query, height=90, placeholder="Type your trade, financial, or custom document query here...")

    if st.button("🚀 Execute GraphRAG Analysis"):
        # INPUT VALIDATION: Block empty queries!
        if not user_query.strip():
            st.warning("⚠️ Please enter a query or select a pre-configured scenario before running analysis.")
        else:
            with st.spinner("Retrieving matched facts from Neo4j Knowledge Graph & reasoning with Gemini..."):
                try:
                    report, retrieved_facts = execute_smart_graphrag(user_query)
                    st.success("Analysis Complete — Grounded in Neo4j Knowledge Graph")
                    
                    st.markdown(report)
                    
                    st.write("---")
                    st.markdown("#### 📥 Download Advisory Report")
                    d_col1, d_col2, d_col3 = st.columns(3)
                    
                    with d_col1:
                        st.download_button(
                            label="📄 Download as Markdown (.md)",
                            data=report,
                            file_name="GraphAI_Advisory_Report.md",
                            mime="text/markdown"
                        )
                    with d_col2:
                        export_data = {
                            "user_query": user_query,
                            "advisory_report": report,
                            "retrieved_neo4j_facts": retrieved_facts
                        }
                        st.download_button(
                            label="📊 Download as JSON (.json)",
                            data=json.dumps(export_data, indent=2),
                            file_name="GraphAI_Advisory_Report.json",
                            mime="application/json"
                        )
                    with d_col3:
                        st.download_button(
                            label="📝 Download as Text (.txt)",
                            data=report,
                            file_name="GraphAI_Advisory_Report.txt",
                            mime="text/plain"
                        )
                    
                    with st.expander("🔍 View Retracted Neo4j Source Facts (Audit Trail)"):
                        st.caption("The response above was generated strictly from the following retrieved graph relationships:")
                        for fact in retrieved_facts[:20]:
                            st.markdown(f"`{fact}`")
                            
                except Exception as e:
                    st.error(f"Execution Error: {e}")

# TAB 2: MULTI-FORMAT DOCUMENT INGESTION
with tab2:
    st.markdown("### 📄 Multi-Format Document Ingestion Engine")
    st.write("Upload ANY document format (**PDF**, **DOCX**, **CSV**, **TXT**, or **MD**) to automatically extract Knowledge Graph entities into Neo4j.")
    
    uploaded_file = st.file_uploader("Upload Document (PDF, DOCX, CSV, TXT, MD)", type=["pdf", "docx", "csv", "txt", "md"])
    
    if uploaded_file is not None:
        file_text, file_type = extract_text_from_file(uploaded_file)
        st.info(f"File uploaded: '{uploaded_file.name}' (Format: {file_type.upper()}, Size: {round(uploaded_file.size / 1024, 1)} KB). Click below to parse into Neo4j triples.")
        
        if st.button("⚡ Parse & Ingest Document into Neo4j"):
            with st.spinner(f"Parsing {file_type.upper()}, extracting entities with Gemini, and saving to Neo4j..."):
                try:
                    if file_type == "pdf" and len(file_text) < 100:
                        st.warning("⚠️ Scanned Image PDF detected. Uploading to Gemini Multimodal Engine for OCR & Graph Extraction...")
                        client = get_gemini_client()
                        file_ref = client.files.upload(file="temp_ingest.pdf")
                        
                        prompt = """
                        Extract all key structured entities and relationships from this PDF document into a JSON object matching:
                        {
                          "nodes": [{"id": "EntityName", "label": "EntityType"}],
                          "relationships": [{"source": "EntityA", "type": "RELATIONSHIP_TYPE", "target": "EntityB"}]
                        }
                        Return ONLY valid JSON without code blocks.
                        """
                        chat = client.chats.create(model='gemini-3.6-flash')
                        response = chat.send_message([file_ref, prompt])
                        clean_json = response.text.replace("```json", "").replace("```", "").strip()
                        extracted_json = json.loads(clean_json)
                        st.success("✅ Scanned PDF parsed via Gemini Vision OCR!")
                    else:
                        st.info(f"{file_type.upper()} text extracted ({len(file_text)} characters). Processing with Gemini...")
                        extracted_json = extract_universal_knowledge_graph(file_text[:4000])
                        st.success(f"✅ {file_type.upper()} parsed successfully!")

                    save_universal_triples_to_neo4j(extracted_json)
                    st.success("🎉 Knowledge Graph nodes & relationships successfully saved to Neo4j Cloud!")
                    
                    with st.expander("📊 View Extracted Knowledge Triples"):
                        st.json(extracted_json)
                        
                except Exception as e:
                    st.error(f"Ingestion error: {e}")
                    
                finally:
                    cleanup_temp_files()

# TAB 3: GRAPH INSPECTOR & DATABASE PURGING
with tab3:
    st.markdown("### 📊 Active Knowledge Graph Statistics")
    st.write("Current entity labels loaded in Neo4j Aura Cloud:")
    
    try:
        with driver.session() as session:
            count_result = session.run("MATCH (n) RETURN labels(n)[0] AS Label, count(n) AS Count ORDER BY Count DESC")
            records = list(count_result)
            if records:
                for record in records:
                    st.markdown(f"- **{record['Label']} Nodes:** `{record['Count']}`")
            else:
                st.info("Database is currently empty.")
    except Exception as e:
        st.error(f"Could not load graph metrics: {e}")
        
    st.write("---")
    st.markdown("### 🗑️ Database Management & Data Purging")
    st.caption("Remove ingested knowledge or reset your Neo4j Cloud instance.")
    
    col_del1, col_del2 = st.columns(2)
    
    with col_del1:
        doc_to_delete = st.text_input("Enter Document Name to Delete (e.g. bank_statement.pdf):")
        if st.button("❌ Delete Specific Document Nodes"):
            if doc_to_delete:
                with driver.session() as session:
                    session.run("MATCH (n {source_doc: $doc_name}) DETACH DELETE n", doc_name=doc_to_delete)
                st.success(f"Deleted all nodes associated with '{doc_to_delete}'!")
                st.rerun()
            else:
                st.warning("Please enter a document name to delete.")
                
    with col_del2:
        st.markdown("**Reset Database**")
        if st.button("⚠️ Clear Entire Neo4j Database"):
            with driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            st.success("🎉 Entire Neo4j Database successfully wiped clean!")
            st.rerun()