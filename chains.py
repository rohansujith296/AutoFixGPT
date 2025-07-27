# feedback_agent.py

import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import Together
from langchain.chains import LLMChain
import sentence_transformers

# ✅ Load environment variables
load_dotenv()

# ✅ Load the embedding model and vectorstore
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectorstore = FAISS.load_local("price_vector_store", embedding, allow_dangerous_deserialization=True)

# ✅ Load the LLM
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.7,
    max_tokens=512,
    together_api_key=os.getenv("TOGETHER_API_KEY")
)

# ✅ Define custom prompt with 'damaged_part' and 'context' from vector DB
template = """
You are AutoFix Assistant. A car has damage to its {damaged_part}.
Based on the following repair price knowledge base, estimate the average repair cost.
Also briefly explain what this part does and the implications of its damage.

## Knowledge Base:
{context}

Respond in this format:
- 📍 Damaged Part: {damaged_part}
- 🧾 Explanation:
- 💸 Estimated Repair Cost:
"""

prompt = PromptTemplate(
    input_variables=["damaged_part", "context"],
    template=template
)

# ✅ Set up the LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# ✅ Final callable function
def get_damage_feedback(damaged_part: str):
    query = f"repair cost for {damaged_part}"
    
    # Perform similarity search in FAISS vector DB
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Run the LLMChain with both prompt inputs
    return chain.run(damaged_part=damaged_part, context=context)
