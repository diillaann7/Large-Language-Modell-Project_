# Comparing The Effectiveness Of RAG Between Models

This university project analyzes how different Large Language Models (LLMs) respond to Retrieval-Augmented Generation (RAG). We based our implementation on the paper "RAGAS: Automated Evaluation of Retrieval Augmented Generation" by Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert.

### Evaluation Metrics
To evaluate the RAG performance, we implemented three scores:
* **Faithfulness**: Does the model use the given context?
* **Answer Relevance**: Does the answer actually address the query?
* **Context Relevance**: Is the retrieved context useful for answering the query?

### Models used
* **Llama 3.1-8B-Instruct**.
* **Ministral-8B-Instruct-2410**
* **Qwen3-8B**

## Built with Llama
This project utilizes **Llama 3.1-8B-Instruct**. 
Llama 3.1 is licensed under the [Llama 3.1 Community License](Llama-LICENSE), Copyright © Meta Platforms, Inc. All Rights Reserved.

### Prerequisites
* **Python**: 3.12.10
* **Hugging Face Access Token**: Required to authenticate and download the Llama 3.1 weights.
* **Environment**: Ensure you have accepted the license terms on the [Hugging Face Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

### How to run the project
* Create a .env file in the root directory and add your Hugging Face Access Token using the following format: HF_TOKEN="[your token here]"
* Create a .venv inside the root directory with the terminal command python -m venv .venv 
* Actiate the virtual enviroment with .venv\Scripts\activate (Windows) or source .venv/bin/activate (Mac/Linux)
* Run pip install -r requirements.txt
* Open rag.ipynb in your IDE (like VS Code or Jupyter Lab), ensure the .venv is selected as your kernel, and run the cells.