# IMDb Movie Genre Classification: A Multi-Model Comparison Pipeline

## Project Overview
This project addresses a core challenge for ad-supported streaming platforms : **Automated Metadata Enrichment**. By leveraging a dataset of over 50,000 movie descriptions, I built a pipeline to classify content into 27 genres. This system demonstrates how to balance inference latency with semantic accuracy by comparing a high-speed linear baseline against a fine-tuned Transformer.  
Project Walkthrough video available on [Youtube](https://youtu.be/GBjJkYTjUzE)

## Performance Results
The pipeline achieves a significant uplift over the baseline while maintaining high generalization (zero data leakage).

| Metric | Baseline (TF-IDF + LR) | DistilBERT (Transformer) |
| :--- | :--- | :--- |
| **Internal Val Accuracy** | ~58.0% | **~65.6%** |
| **Final Test Accuracy** | ~57.7% | **~65.3%** |

## Key Design Choices & System Architecture
1.  **Three-Phase Pipeline:**
    *   **Phase 1 (Baseline):** Establishes a performance floor using TF-IDF and Logistic Regression.
    *   **Phase 2 (Fine-Tuning):** Fine-tunes a `distilbert-base-uncased` model. I chose DistilBERT for its optimal ROI—providing 97% of BERT’s performance with 40% less memory and faster inference.
    *   **Phase 3 (Evaluation):** Conducts a final "blind" evaluation against a hidden solution dataset to measure real-world performance.
2.  **Hardware Optimization:** The code is optimized for **Apple Silicon (MPS backend)**, enabling GPU-accelerated training on macOS.
3.  **Analytical Rigor:** I implemented an internal 80/20 split and monitored both Training and Validation loss to detect overfitting before final testing.
4.  **Data Engineering:** Developed custom parsing logic to handle the IMDb dataset’s non-standard ` ::: ` delimiter and variable-length synopses.

## How to Run
1.  **Prepare Data:** Place `train_data.txt`, `test_data.txt`, and `test_data_solution.txt` in the `/data` directory. 
    *   Dataset source: [IMDb Genre Classification (Kaggle)](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)
2.  **Install Dependencies:** 
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Pipeline:** 
    ```bash
    python src/train_eval.py
    ```

## Attribution & Requirements
*   **Authored by me:** All custom data loading logic (`data_loader.py`), model pipeline architecture, training loops, and evaluation metrics scripts.
*   **Third-party libraries:** Pre-trained transformer weights and the Trainer API are provided by the [HuggingFace Transformers library](https://huggingface.co/docs/transformers/index).


## Trade-offs & Future Improvements
*   **Current Trade-off:** Used a 128-token context window to prioritize training speed on local hardware. 
*   **Future Improvement:** Implement **Multi-label Classification** (e.g., a movie being both 'Horror' and 'Comedy') and utilize **Loss Weighting** to address the class imbalance in minority genres like 'Musical' or 'Short'.