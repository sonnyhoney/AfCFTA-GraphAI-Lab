import os
import glob
import json
import pymupdf  # Modern PyMuPDF import
from neo4j import GraphDatabase
from google import genai
from dotenv import load_dotenv

load_dotenv(override=True)

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+ssc://625bc80b.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ==========================================
# 1. EXTRACT TEXT FROM PDF
# ==========================================
def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    return full_text

# ==========================================
# 2. USE GEMINI TO EXTRACT GRAPH KNOWLEDGE
# ==========================================
def extract_knowledge_graph_from_text(text_chunk):
    prompt = f"""
    Analyze the following AfCFTA trade document text and extract structured graph entities and relationships.

    DOCUMENT TEXT:
    {text_chunk}

    Extract the facts into JSON format with the following structure:
    {{
      "nodes": [
        {{"id": "CountryName", "label": "Country"}},
        {{"id": "ProductName", "label": "Product", "hs_code": "CodeIfAvailable"}},
        {{"id": "PolicyName", "label": "Policy", "duty_rate": "RateIfAvailable"}},
        {{"id": "DocumentName", "label": "Requirement"}},
        {{"id": "AgencyName", "label": "RegulatoryBody"}}
      ],
      "relationships": [
        {{"source": "CountryA", "type": "EXPORTS", "target": "ProductName"}},
        {{"source": "ProductName", "type": "IMPORTED_BY", "target": "CountryB"}},
        {{"source": "ProductName", "type": "GOVERNED_BY", "target": "PolicyName"}},
        {{"source": "ProductName", "type": "REQUIRES_DOCUMENT", "target": "DocumentName"}},
        {{"source": "AgencyName", "type": "INSPECTS_AND_APPROVES", "target": "ProductName"}}
      ]
    }}

    Return ONLY a valid JSON object. Do not include markdown code block syntax or extra text.
    """
    
    chat = client.chats.create(model='gemini-3.6-flash')
    response = chat.send_message(prompt)
    
    # Clean JSON output
    clean_json = response.text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean_json)

# ==========================================
# 3. SAVE EXTRACTED TRIPLES TO NEO4J
# ==========================================
def save_triples_to_neo4j(graph_data):
    with driver.session() as session:
        # Create Nodes
        for node in graph_data.get("nodes", []):
            label = node.get("label", "Entity")
            node_id = node.get("id")
            if not node_id:
                continue
                
            if label == "Product":
                session.run(f"MERGE (n:{label} {{name: $name}}) SET n.hs_code = $hs_code", 
                            name=node_id, hs_code=node.get("hs_code", "N/A"))
            elif label == "Policy":
                session.run(f"MERGE (n:{label} {{name: $name}}) SET n.duty_rate = $duty_rate", 
                            name=node_id, duty_rate=node.get("duty_rate", "N/A"))
            else:
                session.run(f"MERGE (n:{label} {{name: $name}})", name=node_id)
                
        # Create Relationships
        for rel in graph_data.get("relationships", []):
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type", "CONNECTED_TO")
            
            cypher = f"""
            MATCH (a) WHERE a.name = $source
            MATCH (b) WHERE b.name = $target
            MERGE (a)-[r:{rel_type}]->(b)
            """
            session.run(cypher, source=source, target=target)

# ==========================================
# 4. MAIN INGESTION FOR ALL PDFs
# ==========================================
if __name__ == "__main__":
    # Scan current folder for ALL .pdf files
    pdf_files = glob.glob("*.pdf")
    
    if not pdf_files:
        print("⚠️ No PDF files found in the project folder!")
    else:
        print(f"📚 Found {len(pdf_files)} PDF file(s): {pdf_files}\n")
        
        for file_idx, pdf_path in enumerate(pdf_files):
            print(f"==================================================")
            print(f"📄 [{file_idx + 1}/{len(pdf_files)}] Processing: {pdf_path}")
            print(f"==================================================")
            
            try:
                pdf_text = extract_text_from_pdf(pdf_path)
                
                # Split text into chunks
                chunk_size = 3000
                chunks = [pdf_text[i:i+chunk_size] for i in range(0, len(pdf_text), chunk_size)]
                
                print(f"🤖 Extracted text ({len(chunks)} chunks). Processing with Gemini...")
                for idx, chunk in enumerate(chunks[:5]):  # Process first 5 chunks per PDF
                    print(f"  └─ Chunk {idx + 1}/{min(len(chunks), 5)}...")
                    try:
                        extracted_data = extract_knowledge_graph_from_text(chunk)
                        save_triples_to_neo4j(extracted_data)
                    except Exception as e:
                        print(f"     ⚠️ Error processing chunk {idx + 1}: {e}")
                        
                print(f"✅ Successfully ingested '{pdf_path}' into Neo4j!\n")
                
            except Exception as e:
                print(f"❌ Could not process file '{pdf_path}': {e}\n")
                
        print("🎉 ALL PDFs Successfully Ingested into your Neo4j Knowledge Graph!")
        driver.close()