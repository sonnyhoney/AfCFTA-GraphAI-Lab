import os
from neo4j import GraphDatabase
from google import genai

# ==========================================
# 1. CONFIGURATION
# ==========================================

from dotenv import load_dotenv
load_dotenv(override=True)

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+ssc://51204372.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "51204372")  # <--- Updated username fallback
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Google GenAI Client
client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# 2. POPULATE GRAPHS (AfCFTA Trade & Regulations)
# ==========================================
def populate_afcfta_graph(driver):
    cypher_query = """
    // Create African Countries
    MERGE (ng:Country {name: 'Nigeria', region: 'West Africa'})
    MERGE (gh:Country {name: 'Ghana', region: 'West Africa'})
    MERGE (ke:Country {name: 'Kenya', region: 'East Africa'})
    MERGE (rw:Country {name: 'Rwanda', region: 'East Africa'})

    // Create Commodities & Products
    MERGE (cocoa:Product {name: 'Processed Cocoa Powder', hs_code: '1806.10'})
    MERGE (tea:Product {name: 'Black Tea', hs_code: '0902.30'})
    MERGE (textiles:Product {name: 'Cotton Fabrics', hs_code: '5208.11'})

    // Create AfCFTA Trade & Regulatory Protocols
    MERGE (afcfta_gti:Policy {name: 'AfCFTA Guided Trade Initiative', duty_rate: '0%', status: 'Active'})
    MERGE (cert_origin:Requirement {name: 'AfCFTA Certificate of Origin', issuer: 'Chamber of Commerce'})
    MERGE (phytosanitary:Requirement {name: 'Phytosanitary Health Certificate', issuer: 'Ministry of Agriculture'})

    // Create Regulatory Bodies
    MERGE (nafdac:RegulatoryBody {name: 'NAFDAC Nigeria', country: 'Nigeria'})
    MERGE (fda_gh:RegulatoryBody {name: 'FDA Ghana', country: 'Ghana'})

    // Create Trade Relationships & Regulations
    MERGE (gh)-[:EXPORTS {volume_tonnes: 5000}]->(cocoa)
    MERGE (cocoa)-[:IMPORTED_BY]->(ng)
    MERGE (cocoa)-[:GOVERNED_BY]->(afcfta_gti)
    MERGE (cocoa)-[:REQUIRES_DOCUMENT]->(cert_origin)
    MERGE (cocoa)-[:REQUIRES_DOCUMENT]->(phytosanitary)
    MERGE (nafdac)-[:INSPECTS_AND_APPROVES]->(cocoa)

    MERGE (ke)-[:EXPORTS {volume_tonnes: 8000}]->(tea)
    MERGE (tea)-[:IMPORTED_BY]->(gh)
    MERGE (tea)-[:GOVERNED_BY]->(afcfta_gti)
    MERGE (fda_gh)-[:INSPECTS_AND_APPROVES]->(tea)
    """
    with driver.session() as session:
        session.run(cypher_query)
    print("✅ Neo4j Database successfully populated with AfCFTA Trade & Regulatory Data!")

# ==========================================
# 3. GRAPHRAG RETRIEVAL & GENERATION
# ==========================================
def afcfta_graph_rag(driver, user_question):
    # Step A: Query multi-hop trade relationships from Neo4j
    cypher_retrieval = """
    MATCH (exporter:Country)-[r1:EXPORTS]->(prod:Product)-[r2:IMPORTED_BY]->(importer:Country)
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
            fact = (f"Exporting '{record['Product']}' (HS Code: {record['HSCode']}) from {record['Exporter']} to {record['Importer']}: "
                    f"Governed by '{record['Policy']}' (Duty Rate: {record['DutyRate']}). "
                    f"Required Documents: [{docs}]. Approved/Inspected by: {record['Regulator']}.")
            trade_context.append(fact)
            
    context_str = "\n".join(trade_context)
    
    # Step B: Construct Gemini Prompt
    prompt = f"""
    You are an expert AfCFTA Cross-Border Trade & Regulatory AI Advisor built for the Google Africa Applied AI Lab.
    Answer the user's question accurately using ONLY the structured Graph Database context retrieved from Neo4j.

    --- AfCFTA TRADE KNOWLEDGE GRAPH CONTEXT ---
    {context_str}
    -------------------------------------------

    User Question: {user_question}

    Instructions: Provide a clear, professional trade advisory response explaining the export path, applicable tariff/duty rates under AfCFTA, required documentation, and regulatory compliance bodies.
    """
    
    # Step C: Use Chat API with gemini-3.6-flash 
    chat = client.chats.create(model='gemini-3.6-flash')
    response = chat.send_message(prompt)
    return response.text

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Connect to Neo4j using neo4j+ssc://
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        # 1. Populate the database
        populate_afcfta_graph(driver)
        
        # 2. Test User Question
        question = "What are the rules, required documents, and regulatory approvals for exporting Processed Cocoa Powder from Ghana to Nigeria under AfCFTA?"
        
        print(f"\n❓ Question: {question}\n")
        print("🤖 Querying AfCFTA Knowledge Graph + Google Gemini...\n")
        
        # 3. Execute GraphRAG
        answer = afcfta_graph_rag(driver, question)
        
        print("================ AfCFTA AI ADVISOR RESPONSE ================")
        print(answer)
        print("===========================================================")
        
    finally:
        driver.close()