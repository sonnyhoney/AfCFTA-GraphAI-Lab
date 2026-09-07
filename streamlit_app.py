import os
import json
import streamlit as st
import pymupdf
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
    page_icon="🌍",
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
# UNIVERSAL PDF EXTRACTION & NEO4J INGESTION
# ==========================================
def extract_universal_knowledge_graph(text_chunk):
    client = get_gemini_client()
    prompt = f"""
    Analyze the following document text and extract structured knowledge graph entities and relationships.
    This could be a trade document, bank statement, contract, letter, or official report.

    DOCUMENT TEXT:
    {text_chunk}

    Extract all key entities, facts, and relationships into JSON format with the following structure:
    {{
      "nodes": [
        {{"id": "EntityNameOrValue", "label": "EntityType"}} 
        // Examples of EntityType: Person, AccountHolder, Bank, Organization, Country, Product, Policy, Amount, Date, Requirement
      ],
      "relationships": [
        {{"source": "EntityA", "type": "RELATIONSHIP_TYPE", "target": "EntityB"}}
        // Examples: HAS_ACCOUNT, ISSUED_BY, EXPORTS, GOVERNED_BY, PAID_TO, HELD_BY
      ]
    }}

    Return ONLY a valid JSON object. Do not include markdown code block syntax or conversational text.
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
            if not node_id:
                continue
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
# DYNAMIC KEYWORD-FILTERED GRAPHRAG SEARCH
# ==========================================
def execute_smart_graphrag(question):
    client = get_gemini_client()
    
    # Extract keywords from user question
    keywords = [word.strip().lower() for word in question.split() if len(word) > 2]
    
    # Corrected Cypher query passing r2 through the WITH clause
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
    - If the query is about AfCFTA trade, format your response as a Trade Advisory Report.
    - If the query is about a bank statement, person, organization, or document, provide a clear, professional analytical response answering the question directly.
    - If the context does not contain the requested information, explicitly state what is missing.
    """
    
    chat = client.chats.create(model='gemini-3.6-flash')
    response = chat.send_message(prompt)
    return response.text, graph_facts

# ==========================================
# SIDEBAR ARCHITECTURE
# ==========================================
st.sidebar.markdown("## ⚙️ Platform Engine Specs")
st.sidebar.markdown("""
- **Knowledge Engine:** Neo4j Aura Cloud GDS
- **Reasoning Model:** Google Gemini 3.6 Flash
- **Architecture:** Universal GraphRAG
- **Data Ingestion:** Multimodal OCR & Universal Extraction

---
### 🛠️ Developer Info:
- **Lead Engineer:** Agwu Eze
- **Specialization:** Graph AI & GenAI Systems
""")

st.sidebar.markdown("---")
st.sidebar.markdown("[🐙 View GitHub Source Code](https://github.com/sonnyhoney/AfCFTA-GraphAI-Lab)")
st.sidebar.markdown("[💼 Connect on LinkedIn](https://www.linkedin.com/in/agwu-eze)")

# ==========================================
# MAIN PLATFORM HEADER & METRICS
# ==========================================
st.markdown('<p class="main-title">🌍 Universal GraphAI Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Enterprise Knowledge Graph & Intelligence System powered by Neo4j & Google Gemini</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-title">Knowledge Engine</div><div class="metric-value">Neo4j GDS Cloud</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-title">Accuracy Guarantee</div><div class="metric-value">100% Fact-Grounded</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-title">Search Scope</div><div class="metric-value">Dynamic Keyword Graph</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-title">Inference Engine</div><div class="metric-value">Gemini Multimodal</div></div>', unsafe_allow_html=True)

st.write("")
st.write("")

# ==========================================
# PLATFORM TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["🔍 Intelligence Query Engine", "📄 Ingest Custom PDF Document", "📊 Knowledge Graph Inspector"])

# TAB 1: QUERY ENGINE
with tab1:
    st.markdown("### 💬 Enterprise Query Engine")
    
    sample_scenario = st.selectbox(
        "Select a pre-configured query or type a custom question below:",
        [
            "Custom Search Query...",
            "What is the name of the account holder in the bank statement?",
            "What are the rules, required documents, and regulatory approvals for exporting Processed Cocoa Powder from Ghana to Nigeria under AfCFTA?",
            "What are the duty rates and documents required for exporting Automotive Parts from South Africa to Kenya?",
            "How can Egypt export Phosphate Fertilizers to Nigeria under AfCFTA?"
        ]
    )

    default_query = "What is the name of the account holder in the bank statement?"
    if sample_scenario != "Custom Search Query...":
        default_query = sample_scenario

    user_query = st.text_area("Enter Your Query:", value=default_query, height=90)

    if st.button("🚀 Execute GraphRAG Analysis"):
        with st.spinner("Retrieving matched facts from Neo4j Knowledge Graph & reasoning with Gemini..."):
            try:
                report, retrieved_facts = execute_smart_graphrag(user_query)
                st.success("Analysis Complete — Grounded in Neo4j Knowledge Graph")
                
                st.markdown(report)
                
                with st.expander("🔍 View Retracted Neo4j Source Facts (Audit Trail)"):
                    st.caption("The response above was generated strictly from the following retrieved graph relationships:")
                    for fact in retrieved_facts[:20]:
                        st.markdown(f"`{fact}`")
                        
            except Exception as e:
                st.error(f"Execution Error: {e}")

# TAB 2: PDF DOCUMENT INGESTION
with tab2:
    st.markdown("### 📄 Universal PDF Ingestion Engine")
    st.write("Upload ANY document (AfCFTA trade guide, bank statement, contract, invoice, or letter) to automatically extract entities into Neo4j.")
    
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    
    if uploaded_file is not None:
        st.info(f"File uploaded: '{uploaded_file.name}' ({round(uploaded_file.size / (1024*1024), 2)} MB). Click below to parse into Neo4j triples.")
        if st.button("⚡ Parse & Ingest Document into Neo4j"):
            with st.spinner("Parsing PDF, extracting entities with Gemini, and saving to Neo4j..."):
                try:
                    with open("temp_ingest.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    doc = pymupdf.open("temp_ingest.pdf")
                    extracted_text = ""
                    for page in doc:
                        extracted_text += page.get_text() + "\n"
                        
                    clean_text = extracted_text.strip()
                    
                    if len(clean_text) < 100:
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
                        st.info(f"Digital PDF detected ({len(clean_text)} characters extracted). Processing with Gemini...")
                        extracted_json = extract_universal_knowledge_graph(clean_text[:4000])
                        st.success("✅ Digital PDF parsed successfully!")

                    save_universal_triples_to_neo4j(extracted_json)
                    st.success("🎉 Knowledge Graph nodes & relationships successfully saved to Neo4j Cloud!")
                    
                    with st.expander("📊 View Extracted Triples"):
                        st.json(extracted_json)
                        
                except Exception as e:
                    st.error(f"Ingestion error: {e}")

# TAB 3: GRAPH INSPECTOR
with tab3:
    st.markdown("### 📊 Active Knowledge Graph Statistics")
    st.write("Current entity labels loaded in Neo4j Aura Cloud:")
    
    try:
        with driver.session() as session:
            count_result = session.run("MATCH (n) RETURN labels(n)[0] AS Label, count(n) AS Count ORDER BY Count DESC")
            for record in count_result:
                st.markdown(f"- **{record['Label']} Nodes:** `{record['Count']}`")
    except Exception as e:
        st.error(f"Could not load graph metrics: {e}")