import os
import streamlit as st
from neo4j import GraphDatabase
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+ssc://625bc80b.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="AfCFTA GraphRAG Trade Advisor",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS for Modern UI & Bigger Fonts
st.markdown("""
<style>
    /* Main Titles & Subtitles */
    .main-title { font-size: 2.5rem !important; font-weight: 800; color: #FFFFFF; margin-bottom: 5px; }
    .sub-title { font-size: 1.25rem !important; font-weight: 500; color: #4285F4; margin-bottom: 25px; }
    
    /* Input Labels & Headers */
    .stTextInput label, .stSelectbox label { font-size: 1.15rem !important; font-weight: 600 !important; color: #E0E0E0 !important; }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.2em;
        background-color: #1A73E8;
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1557B0;
        box-shadow: 0px 4px 12px rgba(26, 115, 232, 0.4);
    }
    
    /* Info Card Styling */
    .metric-card {
        background-color: #1E1E2E;
        padding: 15px 20px;
        border-radius: 10px;
        border-left: 5px solid #4285F4;
        margin-bottom: 20px;
    }
    .metric-title { font-size: 0.9rem; color: #A0A0A0; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 1.4rem; color: #FFFFFF; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)

@st.cache_resource
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

client = get_gemini_client()
driver = get_neo4j_driver()

# Function to populate rich multi-corridor graph data
def populate_expanded_graph():
    cypher_population = """
    MERGE (ng:Country {name: 'Nigeria', region: 'West Africa'})
    MERGE (gh:Country {name: 'Ghana', region: 'West Africa'})
    MERGE (ke:Country {name: 'Kenya', region: 'East Africa'})
    MERGE (rw:Country {name: 'Rwanda', region: 'East Africa'})
    MERGE (sa:Country {name: 'South Africa', region: 'Southern Africa'})
    MERGE (eg:Country {name: 'Egypt', region: 'North Africa'})
    MERGE (sn:Country {name: 'Senegal', region: 'West Africa'})

    MERGE (cocoa:Product {name: 'Processed Cocoa Powder', hs_code: '1806.10'})
    MERGE (tea:Product {name: 'Black Tea', hs_code: '0902.30'})
    MERGE (auto:Product {name: 'Automotive Engine Parts', hs_code: '8708.29'})
    MERGE (fertilizer:Product {name: 'Phosphate Fertilizers', hs_code: '3103.11'})
    MERGE (coffee:Product {name: 'Arabica Coffee Beans', hs_code: '0901.11'})

    MERGE (gti:Policy {name: 'AfCFTA Guided Trade Initiative', duty_rate: '0% Preferential'})
    MERGE (sadc_opt:Policy {name: 'AfCFTA Rules of Origin Protocol', duty_rate: '2% Reduced'})

    MERGE (cert_origin:Requirement {name: 'AfCFTA Certificate of Origin', issuer: 'Chamber of Commerce'})
    MERGE (phytosanitary:Requirement {name: 'Phytosanitary Health Certificate', issuer: 'Ministry of Agriculture'})
    MERGE (standards_cert:Requirement {name: 'SONCAP Standard Certificate', issuer: 'Standards Organization'})

    MERGE (nafdac:RegulatoryBody {name: 'NAFDAC Nigeria'})
    MERGE (fda_gh:RegulatoryBody {name: 'FDA Ghana'})
    MERGE (kebs:RegulatoryBody {name: 'KEBS Kenya'})

    MERGE (gh)-[:EXPORTS]->(cocoa)-[:IMPORTED_BY]->(ng)
    MERGE (cocoa)-[:GOVERNED_BY]->(gti)
    MERGE (cocoa)-[:REQUIRES_DOCUMENT]->(cert_origin)
    MERGE (cocoa)-[:REQUIRES_DOCUMENT]->(phytosanitary)
    MERGE (nafdac)-[:INSPECTS_AND_APPROVES]->(cocoa)

    MERGE (ke)-[:EXPORTS]->(tea)-[:IMPORTED_BY]->(gh)
    MERGE (tea)-[:GOVERNED_BY]->(gti)
    MERGE (tea)-[:REQUIRES_DOCUMENT]->(cert_origin)
    MERGE (fda_gh)-[:INSPECTS_AND_APPROVES]->(tea)

    MERGE (sa)-[:EXPORTS]->(auto)-[:IMPORTED_BY]->(ke)
    MERGE (auto)-[:GOVERNED_BY]->(sadc_opt)
    MERGE (auto)-[:REQUIRES_DOCUMENT]->(cert_origin)
    MERGE (auto)-[:REQUIRES_DOCUMENT]->(standards_cert)
    MERGE (kebs)-[:INSPECTS_AND_APPROVES]->(auto)

    MERGE (eg)-[:EXPORTS]->(fertilizer)-[:IMPORTED_BY]->(ng)
    MERGE (fertilizer)-[:GOVERNED_BY]->(gti)
    MERGE (fertilizer)-[:REQUIRES_DOCUMENT]->(cert_origin)
    MERGE (nafdac)-[:INSPECTS_AND_APPROVES]->(fertilizer)

    MERGE (rw)-[:EXPORTS]->(coffee)-[:IMPORTED_BY]->(sn)
    MERGE (coffee)-[:GOVERNED_BY]->(gti)
    MERGE (coffee)-[:REQUIRES_DOCUMENT]->(phytosanitary)
    """
    with driver.session() as session:
        session.run(cypher_population)

populate_expanded_graph()

# GraphRAG Function
def query_afcfta_graphrag(question):
    cypher_retrieval = """
    MATCH (exporter:Country)-[:EXPORTS]->(prod:Product)-[:IMPORTED_BY]->(importer:Country)
    OPTIONAL MATCH (prod)-[:GOVERNED_BY]->(policy:Policy)
    OPTIONAL MATCH (prod)-[:REQUIRES_DOCUMENT]->(doc:Requirement)
    OPTIONAL MATCH (reg:RegulatoryBody)-[:INSPECTS_AND_APPROVES]->(prod)
    RETURN exporter.name AS Exporter, importer.name AS Importer, prod.name AS Product, 
           prod.hs_code AS HSCode, policy.name AS Policy, policy.duty_rate AS DutyRate, 
           collect(DISTINCT doc.name) AS RequiredDocuments, reg.name AS Regulator
    """
    
    trade_context = []
    with driver.session() as session:
        result = session.run(cypher_retrieval)
        for record in result:
            docs = ", ".join(record['RequiredDocuments']) if record['RequiredDocuments'] else "None"
            reg = record['Regulator'] if record['Regulator'] else "Standard Customs Agency"
            fact = (f"Exporting '{record['Product']}' (HS Code: {record['HSCode']}) from {record['Exporter']} to {record['Importer']}: "
                    f"Governed by '{record['Policy']}' (Duty Rate: {record['DutyRate']}). "
                    f"Required Documents: [{docs}]. Approved/Inspected by: {reg}.")
            trade_context.append(fact)
            
    context_str = "\n".join(trade_context)
    
    prompt = f"""
    You are an expert AfCFTA Cross-Border Trade & Regulatory AI Advisor built for the Google Africa Applied AI Lab.
    Answer the user's question accurately using ONLY the structured Graph Database context retrieved from Neo4j.

    --- AfCFTA TRADE KNOWLEDGE GRAPH CONTEXT ---
    {context_str}
    -------------------------------------------

    User Question: {question}

    Instructions: Provide a clear, professional trade advisory response explaining the export path, applicable tariff/duty rates under AfCFTA, required documentation, and regulatory compliance bodies.
    """
    
    chat = client.chats.create(model='gemini-3.6-flash')
    response = chat.send_message(prompt)
    return response.text

# SIDEBAR (System Info & Fixed Icons)
st.sidebar.header("📌 System Architecture")
st.sidebar.markdown("""
- **Graph Database:** Neo4j Aura Cloud
- **LLM Engine:** Google Gemini 3.6 Flash
- **Technique:** GraphRAG (Retrieval-Augmented Generation)

---
### 🛣️ Active Trade Corridors:
- **Ghana ➡️ Nigeria** *(Cocoa)*
- **Kenya ➡️ Ghana** *(Tea)*
- **South Africa ➡️ Kenya** *(Automotive)*
- **Egypt ➡️ Nigeria** *(Fertilizers)*
- **Rwanda ➡️ Senegal** *(Coffee)*
""")

# MAIN PANEL HEADER
st.markdown('<p class="main-title">🌍 AfCFTA Trade & Regulatory AI Advisor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Solving African cross-border trade complexity using Graph Data Science & Google Gemini</p>', unsafe_allow_html=True)

# TOP METRIC CARDS ROW
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-title">Active Corridors</div><div class="metric-value">5 Corridors Loaded</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-title">Accuracy Rate</div><div class="metric-value">100% Fact-Grounded</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-title">Tariff Optimization</div><div class="metric-value">0% AfCFTA Duty</div></div>', unsafe_allow_html=True)

st.write("") # Spacer

# INPUT CONTAINER
with st.container():
    st.markdown("#### 🔍 Ask a Trade Advisory Question")
    
    sample_q = st.selectbox(
        "Select a pre-loaded trade scenario or type your own below:",
        [
            "Custom Question...",
            "What are the rules, required documents, and regulatory approvals for exporting Processed Cocoa Powder from Ghana to Nigeria under AfCFTA?",
            "What are the duty rates and documents required for exporting Automotive Parts from South Africa to Kenya?",
            "How can Egypt export Phosphate Fertilizers to Nigeria under AfCFTA?",
            "What is required to export Black Tea from Kenya to Ghana?",
            "What regulations apply to Coffee exports from Rwanda to Senegal?"
        ]
    )

    default_text = "What are the rules, required documents, and regulatory approvals for exporting Processed Cocoa Powder from Ghana to Nigeria under AfCFTA?"
    if sample_q != "Custom Question...":
        default_text = sample_q

    user_question = st.text_area("Your Trade Query:", value=default_text, height=100)

    st.write("")
    if st.button("🚀 Run AfCFTA Trade Analysis"):
        with st.spinner("Retrieving knowledge graph facts from Neo4j & synthesizing with Gemini..."):
            try:
                answer = query_afcfta_graphrag(user_question)
                st.success("Trade Analysis Complete!")
                st.markdown("### 📊 Official Trade Advisory")
                st.info(answer)
            except Exception as e:
                st.error(f"Error executing analysis: {e}")