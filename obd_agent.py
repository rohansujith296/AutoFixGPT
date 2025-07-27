# agents/obd_feedback_agent.py

import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.llms import Together
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ✅ Load environment variables
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectorstore = FAISS.load_local("vector_store", embedding, allow_dangerous_deserialization=True)

# ✅ LLM config (Mistral on Together AI)
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.7,
    max_tokens=350,
    together_api_key=os.getenv("TOGETHER_API_KEY")
)

# ✅ Custom prompt template
prompt_template = PromptTemplate(
    input_variables=["query", "context"],
    template="""
You are AutoFix-GPT, an expert in diagnosing OBD faults and vehicle warning signs.

Given the following diagnostic trouble code or symptom:
"{query}"

Use the repair manual knowledge below to:
- Explain the root cause of the issue
- Provide possible DIY fixes (if applicable)
- Recommend whether a service center visit is required

## Repair Knowledge Base:
{context}

Respond in this format:
🔍 Fault Explanation: ...
🛠️ Suggested Fixes: ...
🏁 Recommendation: ...
"""
)

# ✅ Build custom LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# ✅ Function to call
def get_obd_feedback(query: str) -> str:
    search_results = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in search_results])
    return llm_chain.run(query=query, context=context)
