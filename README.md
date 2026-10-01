# Text Document Classification

A beginner-friendly educational project that shows how to build a simple **text document classification** system in Python. It uses a **Bag-of-Words** representation (`CountVectorizer`) and a **Multinomial Naive Bayes** classifier from scikit-learn to predict the category of a piece of text.

The entire workflow lives in one Jupyter notebook, so students can read, run, and modify it step by step.

## What You Will Learn

- How labeled text data is structured for supervised learning (`text` → `label`)
- How to split data into training and test sets
- How to convert text into numerical features with Bag-of-Words
- How to train a Multinomial Naive Bayes classifier
- How to evaluate a model with accuracy and a confusion matrix
- How to classify new text typed in by the user

## Project Structure

```
classification-of-text-document/
├── app.ipynb                  # Main notebook: the full workflow, step by step
├── synthetic_text_data.csv    # Small synthetic dataset (columns: text, label)
├── src/
│   └── classification_of_text/  # Python package source
├── pyproject.toml             # Project metadata and configuration
├── uv.lock                    # Locked dependency versions (uv)
├── .python-version            # Python version used for development
└── README.md
```

## Dataset

`synthetic_text_data.csv` is a small synthetic dataset (about 85 rows) created for learning purposes. It has two columns:

| Column  | Description                                   |
| ------- | --------------------------------------------- |
| `text`  | A short news-style sentence                   |
| `label` | The topic category the sentence belongs to    |

Example categories include topics such as *Entertainment*. Because the data is synthetic and small, it is ideal for fast experiments, but it is **not** meant to represent real-world performance.

## Requirements

- Python **3.11** or newer
- Jupyter (Notebook, JupyterLab, or VS Code with the Jupyter extension)
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/MikaeelDay/classification-of-text-document.git
cd classification-of-text-document
```

**2. Install the dependencies**

Using `uv`:

```bash
uv add pandas numpy matplotlib seaborn scikit-learn ipykernel
```

Or using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**3. Launch the notebook**

```bash
jupyter notebook app.ipynb
```

Then run the cells from top to bottom.

## How It Works

The notebook follows a standard text classification pipeline:

1. **Load the data** from `synthetic_text_data.csv` with pandas
2. **Split** the data into training (80%) and test (20%) sets using `train_test_split` (`random_state=42`)
3. **Vectorize** the text with `CountVectorizer`, which turns each document into a vector of word counts (fitted on the training set only, then applied to the test set)
4. **Train** a `MultinomialNB` (Naive Bayes) classifier on the training vectors
5. **Predict** the labels of the test set
6. **Evaluate** the model with accuracy and a confusion matrix, visualized as a seaborn heatmap
7. **Classify new text**: the last cell asks for any sentence and prints the predicted category

### Example

```
Input Your Text : i watched a good movie today named spiderman2
The input text belongs to the 'Entertainment' category.
```

## Results

On the 20% test split, the model reaches an accuracy of about **88%**. Since the dataset is tiny, the result can change noticeably with a different split or more data.

## Ideas for Practice

- Add more examples and new labels to the dataset
- Replace `CountVectorizer` with `TfidfVectorizer` and compare the accuracy
- Try other classifiers such as Logistic Regression or a linear SVM
- Remove stop words or lowercase and clean the text, then measure the effect
- Use cross-validation instead of a single train/test split
- Replace the synthetic data with a real-world dataset

## Contributing

This is a learning project, and suggestions are welcome. Feel free to open an issue or submit a pull request.

## Author

**Mikaeel Raeisi** — [@MikaeelDay](https://github.com/MikaeelDay)
