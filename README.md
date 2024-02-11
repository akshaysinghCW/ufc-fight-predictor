# UFC Fight Predictor Project

## Overview
The UFC Fight Predictor is an innovative tool designed to forecast the outcomes of UFC fights with impressive accuracy. Utilizing state-of-the-art web scraping techniques, meticulous data processing, and advanced machine learning models, this project offers enthusiasts and analysts a unique perspective on predicting fight results.

## Features
- **Data Acquisition**: Leverages BeautifulSoup for efficient web scraping, gathering comprehensive fight data directly from relevant UFC sources.
- **Data Processing**: Employs a manual cleaning and processing approach within Jupyter Notebooks, ensuring data quality and relevance for model training.
- **Machine Learning Models**: Tests various predictive models, with Random Forest emerging as the top performer due to its superior accuracy in forecasting fight outcomes.
- **User Interface**: Features a user-friendly Streamlit application, allowing users to interactively select fighters and view predicted match outcomes.

## Getting Started

### Prerequisites
- Python 3.6+
- BeautifulSoup4
- Pandas
- Scikit-learn
- Streamlit
- Jupyter Notebook

### Usage
1. Launch the Jupyter Notebook to explore data cleaning and processing:
   ```
   jupyter notebook data_processing.ipynb
   ```
2. Train the machine learning models and evaluate their performance within the notebook.
3. Run the Streamlit app to interact with the fight predictor:
   ```
   streamlit run app.py
   ```

## Methodology
- **Data Scraping**: Utilized BeautifulSoup to scrape fight data, including fighter statistics, past performances, and match outcomes.
- **Data Cleaning**: Manually cleaned and processed the scraped data in Jupyter Notebook to ensure quality and consistency for model training.
- **Model Selection**: Experimented with various machine learning models, including Logistic Regression, Decision Trees, and SVM. Random Forest was selected for its high accuracy and robustness.
- **Deployment**: The final model was deployed in a Streamlit application, providing an intuitive interface for users to make predictions.

## Results
The Random Forest model achieved the highest accuracy, making it the backbone of our UFC Fight Predictor. This model's success lies in its ability to handle the complex nature of fight data and its proficiency in capturing the intricate relationships between various fighting attributes.

## Connect
Interested in discussing this project further or collaborating on similar initiatives? Feel free to reach out and connect with me.

- [LinkedIn](https://www.linkedin.com/in/your-linkedin)
- [GitHub](https://github.com/your-github)

## Acknowledgements
Special thanks to all contributors and the open-source community for the tools and libraries that made this project possible.
