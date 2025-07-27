# 🚗 AutoFix-GPT: AI-Powered Vehicle Damage Diagnosis and Repair Assistant

AutoFix-GPT is an intelligent multi-modal assistant that predicts car damage type from uploaded images, decodes OBD-II error codes, and provides detailed repair feedback and cost estimation using LLMs and vector databases.

> 🔧 Built for automobile enthusiasts, service centers, and insurance assessors.

---

## 🧠 Features

- ✅ **Damage Image Classifier**: Upload a car damage image and predict one of 6 damage types using a CNN-based model.
- 📟 **OBD-II Decoder**: Enter error codes and receive plain-English explanations powered by LangChain + Together LLM.
- 💸 **Repair Cost Estimator**: Get contextual feedback and cost suggestions using a custom price knowledge vector DB.
- 🧾 **PDF-based Knowledge RAG**: Embeds a dataset of repair/service costs for accurate LLM retrieval.
- 🌐 **Streamlit UI**: Clean and interactive interface for image uploads, code inputs, and real-time feedback.


---

## 🧱 Tech Stack

| Layer            | Tools/Packages                                                                 |
|------------------|---------------------------------------------------------------------------------|
| 👁️ Image Model   | TensorFlow 2.11, Keras 2.11 (trained on 224x224 damage images)                  |
| 🔤 Text Analysis | LangChain 0.1.16 + Together AI LLM (Mistral-7B-Instruct via API)                |
| 🔍 Embeddings    | HuggingFaceEmbeddings or TensorflowHub USE (fallback)                          |
| 🧠 Vector DB     | FAISS + custom repair cost PDF embeddings (`price_vector_store/`)              |
| 🖥️ UI            | Streamlit 1.20.0 – responsive multi-tab UI                                     |
| 🔙 Backend       | Optional Flask API integration for future expansion                            |
| 🧪 DevTools      | Python 3.10.3, virtualenv, pydantic==1.10.13                                    |

---

## 🚀 How It Works

```mermaid
graph TD
    A[User Uploads Image / OBD Code] --> B[Streamlit UI]
    B --> C[Damage Classifier or Code Parser]
    C --> D[LLM via LangChain + Together API]
    D --> E[Cost Estimator (FAISS Vector Store)]
    E --> F[Return Feedback + Repair Estimate]

# 📸 Damage Classes
##The image classifier is trained on the following 6 classes:

    Front Bumper

    Rear Bumper

    Hood

    Door

    Windshield

    No Damage

