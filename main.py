import streamlit as st
import pandas as pd
# scikit learn - important - v 1.4.0
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from joblib import load
rf_classifier = load('rf_classifier.joblib')

st.title('UFC Fight Predictor')


# Data caching
# @st.cache_data
def load_data_from_second_url(csv_url_2):
    df2 = pd.read_csv(csv_url_2)
    return df2


csv_url2 = 'https://akshaysinngh.com/UFC2/metadata.csv'
df2 = load_data_from_second_url(csv_url2)

# Sidebar
st.sidebar.header('Pick Your Fighters')

# Map weight division names to numeric values
weight_division_map = {
    'Strawweight': 115,
    'Flyweight': 125,
    'Bantamweight': 135,
    'Featherweight': 145,
    'Lightweight': 155,
    'Welterweight': 170,
    'Middleweight': 185,
    'Light Heavyweight': 205,
    'Heavyweight': 265
}

# Inverse mapping for displaying in selectbox
inverse_weight_division_map = {v: k for k, v in weight_division_map.items()}

# Gender selection
default_gender_index = 0  # Default to the first option, e.g., "Men"
gender_category = st.sidebar.selectbox('Gender', ("Men", "Women"), index=default_gender_index)
gender_value = 1 if gender_category == "Men" else 0

# Filter the DataFrame based on the selected gender
filtered_df = df2[df2['Gender'] == (1 if gender_category == "Men" else 0)]

# Get unique weight divisions for the selected gender
unique_weight_divisions = filtered_df['Weight Division'].unique()
weight_divisions = [inverse_weight_division_map[weight] for weight in unique_weight_divisions if weight in inverse_weight_division_map]

default_weight_division = 'Lightweight'  # Example default weight division
if default_weight_division in weight_divisions:
    default_weight_division_index = weight_divisions.index(default_weight_division)
else:
    default_weight_division_index = 0

# Weight division selection
selected_weight_division_name = st.sidebar.selectbox('Weight Division', weight_divisions, index=default_weight_division_index)
selected_weight_division = weight_division_map[selected_weight_division_name]

# Further filter the DataFrame based on the selected weight division
final_filtered_df = filtered_df[filtered_df['Weight Division'] == selected_weight_division]

# Get unique fighter names for the selected gender and weight division
unique_fighters = final_filtered_df['fighter'].unique()

# Fighter selection
default_red_corner = 'Conor McGregor'
default_blue_corner = 'Michael Chandler'

# Set default for blue corner
if default_blue_corner in unique_fighters:
    default_blue_corner_index = list(unique_fighters).index(default_blue_corner)
else:
    default_blue_corner_index = 0  # Fallback to the first item if default is not in the list

# Set default for red corner
if default_red_corner in unique_fighters:
    default_red_corner_index = list(unique_fighters).index(default_red_corner)
else:
    default_red_corner_index = 0  # Fallback to the first item if default is not in the list

red_corner = st.sidebar.selectbox('Red Corner', unique_fighters, index=default_red_corner_index)
blue_corner = st.sidebar.selectbox('Blue Corner', unique_fighters, index=default_blue_corner_index)

# End of Sidebar


@st.cache_data
def load_data_from_first_url(csv_url_1):
    df1 = pd.read_csv(csv_url_1)
    return df1


csv_url_1 = 'https://akshaysinngh.com/UFC2/processed_fight_data.csv'
dataframe = load_data_from_first_url(csv_url_1)


def get_fighter_row(dataframe, fighter_name):
    row = dataframe.loc[dataframe['fighter'] == fighter_name]
    row_values = row.values.flatten()[1:]  # Exclude the 'fighter' column
    return row_values


def get_win_probability(fighter_name):
    fighter_name = fighter_name
    fighter_row = get_fighter_row(dataframe, fighter_name)
    win_prob = rf_classifier.predict_proba([fighter_row])[0][1]
    return win_prob


def match_probability(fighter_A, fighter_B):
    fighter_A = fighter_A
    fighter_B = fighter_B
    prob_A = get_win_probability(fighter_A)
    prob_B = get_win_probability(fighter_B)
    total_prob = prob_A + prob_B
    probability_A = round(prob_A / total_prob, 2)
    probability_B = round(prob_B / total_prob, 2)
    return probability_A, probability_B


blue_prob, red_prob = match_probability(blue_corner, red_corner)


# Function to get the image URL for a given fighter
def get_image_url(dataframe, fighter_name):
    image_url = dataframe.loc[dataframe['fighter'] == fighter_name, 'Image URL'].iloc[0]
    return image_url


# Get the image URLs for the selected fighters
blue_corner_image_url = get_image_url(final_filtered_df, blue_corner)
red_corner_image_url = get_image_url(final_filtered_df, red_corner)


st.write('Predicted Probabilities: Who Would Win in a Hypothetical Matchup?')

col3, col4 = st.columns(2)
col3.image(red_corner_image_url, width=200)
col4.image(blue_corner_image_url, width=200)

col1, col2 = st.columns(2)
col1.subheader(f'{red_corner}:  {red_prob}')
col2.subheader(f'{blue_corner}:  {blue_prob}')


st.markdown("""
This app predicts fight outcomes and win probabilities using Random Forest Machine Learning model.
* **Python libraries used:** sklearn, pandas, streamlit
* **Model Used:** Random Forest
* **Accuracy:** 83% on training data
* **Data source:** [ufcstats.com](https://ufcstats.com)
* **Note:** Prediction accuracy for fighters with less than 3 UFC fights may be lower, due to data being limited.

This model uses the following fight stats to determine match odds.
""")
st.text('knockdowns, significant strikes landed, total strikes landed, takedowns, submission attempts, reversals,'
        'control time, strikes to the head, strikes to the body, strikes to the legs, strikes at distance, '
        'strikes in the clinch, strikes on the ground, win by, weight class, past wins')


st.write('Please refrain from using this app for betting purposes. It was designed as a fun, educational project to '
         'showcase the capabilities of Machine Learning.')

st.markdown('**Made with ❤️ by Akshay Singh**')
st.caption("For more interesting projects, check out my portfolio website."
           "I'm always looking for interesting Data Science projects to work on."
           "Have a brilliant idea? Let's connect and transform it into reality!")
footer1, footer2, footer3 = st.columns(3)
footer1.markdown('<a href="https://akshaysinngh.com" target="_blank">Portfolio Website</a>', unsafe_allow_html=True)
footer2.markdown('<a href="https://linkedin.com/singh-akshay" target="_blank">LinkedIn</a>', unsafe_allow_html=True)
footer3.markdown('<a href="https://github.com" target="_blank">GitHub</a>', unsafe_allow_html=True)

# st.markdown('_Markdown_')
# st.write('Most objects') # df, err, func, keras!
# st.text('Made with ❤️ by Akshay Singh')
