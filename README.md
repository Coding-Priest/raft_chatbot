# Retrieval Augmented Fine-Tuning Project

This repository contains the code and resources for a project focused on retrieval-augmented fine-tuning of a language model. The project leverages synthetic data generation, fine-tuning a Flan-T5 model, and implements a Streamlit frontend for interacting with PDF documents.

## Overview

This project explores the power of retrieval-augmented generation by:

1.  **Synthetic Data Generation:** Creating a custom training dataset using an LLM (Large Language Model). This synthetic dataset is used for the fine-tuning phase.
2.  **Fine-Tuning Flan-T5:** Fine-tuning a Flan-T5 model on the generated synthetic data, enhancing its performance on the specific task.
3.  **Streamlit Frontend:** Building a user-friendly Streamlit interface that allows interaction with PDF documents. This includes retrieval and answering questions based on the content of PDF files.

## Repository Structure

The repository is organized as follows:

*   **Training/**: Contains the core code and datasets for the project.
    *   `Raft_T5.ipynb`: Jupyter notebook with the code for fine-tuning the Flan-T5 model.
    *  `Raft_synthetic_dataset.ipynb`: Jupyter notebook responsible for generating the synthetic dataset.
*   `requirements.txt`: A list of required Python packages to run the project.
*   `runtime.txt`: (Appears to be for environment specification/runtime details - may need updates)
*   `test.py`: A Streamlit application for interacting with the fine-tuned model and PDF documents.

## Project Components

### 1. Synthetic Data Generation
*   **Notebook:** `Raft_synthetic_dataset.ipynb`
*   **Description:** This notebook generates the custom training data using an LLM.
*   **Output:** The resulting dataset is used to fine-tune the Flan-T5 model in the subsequent stage.

### 2. Flan-T5 Fine-Tuning
*   **Notebook:** `Raft_T5.ipynb`
*   **Description:** This notebook fine-tunes the Flan-T5 model using the synthetic data.
*   **Output:** A fine-tuned model capable of performing the specific tasks defined by the training data.

### 3. Streamlit PDF Interface
*   **Script:** `test.py`
*   **Description:** This Python script creates a user-friendly Streamlit interface for interacting with PDF documents.
*   **Features:** 
    *   Upload a PDF document
    *   Query the content of the PDF document
    *   Utilizes the fine-tuned model and retrieval techniques to provide accurate answers.

## Setup and Usage

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2.  **Set up Python environment:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the synthetic dataset generation Notebook:**
    *   Open `Raft_synthetic_dataset.ipynb` in Jupyter Notebook/Lab.
    *   Execute each cell to generate the custom dataset.

4. **Fine-tune Flan-T5:**
    *   Open `Raft_T5.ipynb` in Jupyter Notebook/Lab.
    *   Execute each cell to fine-tune the model using the generated dataset.

5. **Start the Streamlit Application:**
    ```bash
    streamlit run test.py
    ```
6. Access the app via the URL displayed in the console by the streamlit run command.

## Requirements

To run this project, you need the following packages:
*  The dependencies listed in `requirements.txt`

## Additional Information

* The `runtime.txt` file may contain additional information about the run environment. Ensure it is updated for your specific setup if needed.

## Contributing

Contributions to this project are welcome. If you wish to contribute, please follow these steps:

1.  Fork the repository
2.  Create a new branch
3.  Make your changes
4.  Open a Pull Request with a detailed description of the changes made

## License

[Choose a license for your project (e.g. MIT, Apache 2.0). Add here if you have one.]

---

**Key things to customize:**

*   **`<repository_url>`:** Replace with the actual URL of your GitHub repository.
*   **`<repository_directory>`:** Replace with your local repository directory name
*   **License:** Add your project's license information.

**Additional notes:**

*   **Details:** You can add more detail about the specific LLM you used for synthetic data generation (e.g., model name) and any specific configuration used.
*   **Examples:** Add examples of the input/output for the Streamlit app.
*   **Error Handling:** Highlight any potential issues users may encounter and how to resolve them.
*   **Model Checkpoints:** If you are saving model checkpoints, add information about that as well.
*   **Next steps:** You can suggest further research or improvements.

I hope this README is helpful and provides a great start for your project documentation! Let me know if you have any other questions.
