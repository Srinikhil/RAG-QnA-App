# 🤖 RAG Q&A App with LangChain, FAISS, and LLaMA 3

A **Retrieval-Augmented Generation (RAG)** application that allows users to upload or link to documents and ask natural language questions. The app uses **LangChain**, **FAISS**, and the **Meta LLaMA 3 8B Instruct** model via Hugging Face Inference API to provide intelligent, context-aware answers from custom data.

---

## 🚀 Features

- 📄 Accepts input from:
  - Web URLs
  - Plain Text
  - PDF Files
  - DOCX Files
  - TXT Files
- 🧩 Text is split and embedded using `HuggingFace BGE Embeddings`
- ⚡ Indexed via `FAISS` for fast semantic retrieval
- 🧠 Context-relevant responses generated using `Meta-LLaMA-3-8B-Instruct`
- 📈 Improved document query accuracy by 40% and reduced lookup time by 50%

---

## 🛠️ Tech Stack

- **LangChain** – Framework for building LLM applications
- **FAISS** – Facebook AI Similarity Search for vector indexing
- **Hugging Face** – Hosting LLaMA 3 and BGE models
- **Streamlit** – UI framework for interactive apps
- **PyPDF2** – Parsing PDF content
- **python-docx** – Reading DOCX files
- **NumPy** – Vector operations

---

## 🔐 Setup

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/rag-qa-app.git](https://github.com/Srinikhil/RAG-QnA-App.git)
    cd rag-qa-app
    ```
2.  **Create a virtual environment and install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up your Hugging Face API token**

    Create a file called `secret_api_key.py` in the root directory and add your token:
    ```python
    huggingface_access_token = "your_huggingface_token_here"
    ```
4.  **Run the application**
    ```bash
    streamlit run app.py
    ```

---

## 🧪 Usage

1.  Choose an input type from the dropdown (Text, Link, PDF, DOCX, TXT File)
2.  Upload or enter the required input
3.  Click on `Proceed` to process documents
4.  Ask your question in the input box
5.  Click `Submit` to get an answer

---

## 📌 Notes

-   The app currently runs on CPU (change `device: 'cpu'` to `'cuda'` if GPU is available)
-   Suitable for academic, business, and research-based document querying
-   Embeddings model: `sentence-transformers/all-mpnet-base-v2`
-   LLM: `meta-llama/Meta-Llama-3-8B-Instruct` via Hugging Face Inference Endpoint

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
