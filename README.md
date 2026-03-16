# IMDb Movie Genre Classification Pipeline

## Overview
This project addresses the problem of automated metadata enrichment for a streaming content library. It compares two machine learning approaches for classifying movie genres based on their plot descriptions.

## Key Design Choices
1. **Modularity:** The project is divided into dedicated modules for Data Loading, Model Definitions, and Execution, ensuring it is production-ready and maintainable.
2. **Analytical Rigor (Train/Val/Test):** I implemented an internal 80/20 validation split to monitor for overfitting during training. The final evaluation is performed on a completely unseen dataset to ensure the model generalizes well to new content.
3. **Model Selection:** I compared a **Logistic Regression baseline** (for high-speed, low-cost inference) against **DistilBERT** (for deep semantic understanding). DistilBERT was chosen because it provides BERT-level performance at a fraction of the latency and memory footprint, which is essential for large scale.
4. **Data Integrity:** Custom parsing logic was developed to handle the ` ::: ` delimiter used in the IMDb dataset.

## How to Run
1. Place the raw IMDb files (`train_data.txt`, `test_data.txt`, `test_data_solution.txt`) in the `data/` directory. 
2. Install dependencies: `pip install -r requirements.txt`
3. Run the pipeline: `python src/train_eval.py`

## Attribution
- All code in `src/` was authored by me. 
- Pre-trained transformer weights are provided by the [HuggingFace `transformers` library](https://huggingface.co/docs/transformers/models).
- Dataset: IMDb Movie Genre Dataset from [Kaggle](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb?resource=download)