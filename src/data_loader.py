import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class MovieDataLoader:
    """
    Handles loading and parsing of the IMDb Genre dataset using the ' ::: ' separator.
    Includes logic for an internal validation split to monitor overfitting.
    """
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.train_cols = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']
        self.test_cols = ['ID', 'TITLE', 'DESCRIPTION']

    def load_all_data(self, sample_size=None):
        print("--- Loading Data ---")
        # Load files with the specific custom delimiter
        train_df = pd.read_csv("data/train_data.txt", sep=' ::: ', engine='python', names=self.train_cols)
        test_df = pd.read_csv("data/test_data.txt", sep=' ::: ', engine='python', names=self.test_cols)
        solution_df = pd.read_csv("data/test_data_solution.txt", sep=' ::: ', engine='python', names=self.train_cols)

        if sample_size:
            train_df = train_df.sample(n=sample_size, random_state=42)

        # Fit LabelEncoder on training data
        train_df['LABEL'] = self.label_encoder.fit_transform(train_df['GENRE'])
        # Map labels to the test solution ground truth
        solution_df['LABEL'] = self.label_encoder.transform(solution_df['GENRE'])

        # Create an internal 80/20 split from the training data for validation
        train_set, val_set = train_test_split(
            train_df, test_size=0.2, random_state=42, stratify=train_df['LABEL']
        )

        print(f"Dataset Summary: Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_df)}")
        return train_set, val_set, test_df, solution_df, self.label_encoder