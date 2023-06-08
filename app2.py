import streamlit as st
import pandas as pd
import numpy as np
import pickle

#Load the model from the file
with open('ridge_model.pkl', 'rb') as f:
    ridgeRegressor = pickle.load(f)
# Load the scaler from the file
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
# Load the median values from the CSV file
medians = pd.read_csv('medians.csv', index_col=0)
# Set Page configuration
st.set_page_config(page_title='Predict HDB Resale Price', page_icon='🏠', layout='wide', initial_sidebar_state='expanded')
# Set title of the app
st.title('🏠 Predict HDB Resale Price')


# Set input widgets
st.sidebar.subheader('Select HDB attributes')
floor_area_sqm = st.sidebar.slider('Floor Area (sqm)', 0.0, 200.0, medians.loc['floor_area_sqm'].item(), 0.1)
hdb_age = st.sidebar.slider('HDB Age (years)', 0, 100, int(medians.loc['hdb_age'].item()), 1)
max_floor_lvl = st.sidebar.slider('Max Floor Level', 0, 50, int(medians.loc['max_floor_lvl'].item()), 1)
mid = st.sidebar.slider('Mid Floor Level', 0, 50, int(medians.loc['mid'].item()), 1)
which_floor = st.sidebar.slider('Which Floor', 0, 50, int(medians.loc['which_floor'].item()), 1)
mall_within_2km = st.sidebar.slider('Malls within 2km', 0, 10, int(medians.loc['mall_within_2km'].item()), 1)
mrt_name = st.sidebar.slider('Nearest MRT', 0, 10, int(medians.loc['mrt_name'].item()), 1)




selected_col = ['floor_area_sqm', 'pri_sch_name_low','max_floor_lvl', 'flat_model_low', 'sec_sch_name_low', 'hdb_age', 'hawker_within_2km',
'mrt_nearest_distance', 'flat_model_mid', 'mrt_name', 'mid', 'planning_area_low', 'sec_sch_nearest_dist',
'mall_within_2km', 'sqm_year_max_floor', 'max_floor_5room', 'age_3room', 'floor_hawker2km', 'age_totalunit', 'year_floor', 'maxfloor_secsch', 'year_age', 'age_execsold', 'age_pri_sch', 'floor_hawker', 'floor_mall1km', 'storey_range', 'which_floor', 'floor_maxfloor']



# Create a DataFrame for the feature values
df = pd.DataFrame([[floor_area_sqm, hdb_age, max_floor_lvl, mid, which_floor, mall_within_2km, mrt_name]], columns=['floor_area_sqm', 'hdb_age', 'max_floor_lvl', 'mid', 'which_floor', 'mall_within_2km', 'mrt_name'])

# Perform feature engineering
df['sqm_year_max_floor'] = df['floor_area_sqm'] * df['hdb_age'] * df['max_floor_lvl']
df['max_floor_5room'] = df['max_floor_lvl'] * df['floor_area_sqm']  # Assuming floor_area_sqm is proportional to 5room_sold
df['age_3room'] = df['hdb_age'] * df['floor_area_sqm']  # Assuming floor_area_sqm is proportional to 3room_sold
df['floor_hawker2km'] = df['mid'] * df['mall_within_2km']  # Assuming mall_within_2km is proportional to hawker_within_2km
df['age_totalunit'] = df['hdb_age'] * df['floor_area_sqm']  # Assuming floor_area_sqm is proportional to total_dwelling_units
df['year_floor'] = df['hdb_age'] * df['mid']  # Assuming hdb_age is proportional to tranc_year
df['maxfloor_secsch'] = df['max_floor_lvl'] * df['mrt_name']  # Assuming mrt_name is proportional to sec_sch_nearest_dist
df['year_age'] = df['hdb_age'] * df['hdb_age']
df['age_execsold'] = df['hdb_age'] * df['floor_area_sqm']  # Assuming floor_area_sqm is proportional to exec_sold
df['age_pri_sch'] = df['hdb_age'] * df['mrt_name']  # Assuming mrt_name is proportional to pri_sch_nearest_distance
df['floor_hawker'] = df['mid'] * df['mall_within_2km']  # Assuming mall_within_2km is proportional to hawker_market_stalls
df['floor_mall1km'] = df['mid'] * df['mall_within_2km']  # Assuming mall_within_2km is proportional to mall_within_1km
df['floor_maxfloor'] = df['max_floor_lvl'] * df['which_floor']

# Add missing columns with default values
for col in selected_col:
    if col not in df.columns:
        df[col] = medians.loc[col].item()

# Select the columns used for prediction```python
df = df[selected_col]

# Define the columns to be transformed
skew_list_log = ['floor_maxfloor', 'which_floor', 'maxfloor_secsch']
skew_list = ['mall_within_2km', 'age_execsold', 'floor_hawker2km', 'floor_mall1km', 'floor_hawker', 'max_floor_5room']

# Apply log and square root transformations
for col in skew_list_log:
    if col in df.columns:
        df[col] = np.log(df[col])

for col in skew_list:
    if col in df.columns:
        df[col] = np.sqrt(df[col])

# Scale the feature values
df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

# Predict the resale price
predicted_price = ridgeRegressor.predict(df_scaled)




# Display the predicted resale price
st.subheader('Predicted Resale Price')

predicted_price_value = predicted_price.item()  # Extract the value from the ndarray
st.write("SGD$" + "{:,.0f}".format(predicted_price_value))


st.divider()


#run:
#          streamlit run "c:/Users/chaaa/Documents/GitHub/GAProj2/GAProj2/app2.py"
#   streamlit run "/Users/charles/Desktop/GitHub/DSI-SG-37/GAProj2/app2.py"
