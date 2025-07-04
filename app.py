import streamlit as st
import pandas as pd
import joblib

st.title('Personalized travel recommendations')
st.write('This app provides personalized travel recommendations based on your preferences.')
if 'recommendation' not in st.session_state:
    st.session_state.recommendation = ''

st.subheader('Your Recommended Destination category')
if isinstance(st.session_state.recommendation, str):
    st.title(st.session_state.recommendation)

with st.form('predict'):
    #age_group = st.selectbox('Age Group', options=['18-30 years', '31-50 years', '51+ years'], index=1)
    #gender = st.selectbox('Gender', options=['Male', 'Female'], index=0)
    #country = st.selectbox('Country', options=['Sri Lanka', 'USA/Canada', 'Middle East', 'Australia/New Zealand','India','Europe'], index=0)
    #num = st.number_input('Number of people traveling', min_value=1, max_value=10, value=1, step=1)
    travel_group = st.selectbox('Travel Group', options=['Solo traveler', 'Traveling with a friend group', 
                                                         'Traveling with partner', 
                                                         'Traveling with young kids (under 12)', 
                                                         'Traveling with teenagers (12-18)', 
                                                         'Traveling with family (multi-generational)'], index=0)
    budget = st.selectbox('Budget', options=['Budget/Backpacking', 'Mid-range', 'Luxury'], index=0)
    accomodation = st.selectbox('Accommodation', options=['Hostels & guesthouses', 'Budget hotels & Airbnb', '3 - 4 star hotels', '5 - star hotels & luxury resorts'], index=0)
    activity_interest = st.selectbox('Activity Interest', options=['Adventure seeker (hiking, trekking, extreme sports)',
                                                                    'Nature & wildlife lover (safaris, rainforests)',
                                                                    'Beach & water sports enthusiast (surfing, snorkeling, diving)',
                                                                    'Cultural & history enthusiast (temples, heritage sites)',
                                                                    'Spiritual & religious traveler',
                                                                    'Food & culinary explorer',
                                                                    'Photography & scenic views seeker',
                                                                    'Luxury & relaxation traveler',
                                                                    'Business traveler'], index=0)
    physical_activity_level = st.selectbox('Physical Activity Level', options=['Very active (hiking, long walks, adventure sports)',
                                                                               'Moderately active (walking tours, short hikes)', 
                                                                               'Less active (prefer easy access locations, relaxation)'], index=0)
    experience_level = st.selectbox('Experience Level', options=['First-time traveler', 'Have traveled occasionally', 'Frequent traveler'], index=0)
    submit = st.form_submit_button('Get Recommendations')

# The value of st.session_state.recommendation is updated at the end of the script rerun,
# so the displayed value at the top in col2 does not show the new recommendation. Trigger
# a second rerun when the form is submitted to update the value above.
def predict_svm(age_group, gender, country, num=1):
    # Ensure the input is valid
    # Load the model
    model = joblib.load('model.pkl')
    
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Age Group': [age_group],
        'Gender': [gender],
        'Country': [country],
        '': [str(num)]
    })
    label_encoders = joblib.load('label_encoder_dt.pkl')

    for column, encoder in label_encoders.items():
        if column in input_data.columns:
            # Use .transform() to encode the new data
            input_data[column] = encoder.fit_transform(input_data[column])
        else:
            print(f"Warning: Column '{column}' not found in new data. Skipping encoding for this column.")

    scaler = joblib.load('scaler.pkl')
    # Standardize numerical features in sample data
    sample_input_scaled = scaler.fit_transform(input_data) # Use the scaler fitted on the training data

    # Generate prediction using the trained SVM model
    predicted_category_encoded = model.predict(sample_input_scaled)

    # Decode the predicted category back to original words
    predicted_category = label_encoders['Preferred Destination Category'].inverse_transform(predicted_category_encoded)

    return predicted_category

def predict_dt(travel_group, budget, accomodation, activity_interest, physical_activity_level, experience_level):
    # Load the model
    model = joblib.load('random_forest_model.pkl')

    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Travel Group': [travel_group],
        'Budget': [budget],
        'Accommodation': [accomodation],
        'Activity Interest': [activity_interest],
        'Physical Activity Level': [physical_activity_level],
        'Experience Level': [experience_level]
    })

    manual_encoding = {
    "Travel Group": {
        'Solo traveler': 0,
        'Traveling with a friend group': 1,
        'Traveling with partner': 2,
        'Traveling with young kids (under 12)': 3,
        'Traveling with teenagers (12-18)': 4,
        'Traveling with family (multi-generational)': 5
    },
    "Budget": {
        "Budget/Backpacking": 0,
        "Mid-range": 1,
        "Luxury": 2
    },
    "Accommodation": {
        "Hostels & guesthouses": 0,
        "Budget hotels & Airbnb": 1,
        "3 - 4 star hotels": 2,
        "5 - star hotels & luxury resorts": 3
    },
    "Activity Interest": {
        "Adventure seeker (hiking, trekking, extreme sports)": 0,
        "Nature & wildlife lover (safaris, rainforests)": 1,
        "Beach & water sports enthusiast (surfing, snorkeling, diving)": 2,
        "Cultural & history enthusiast (temples, heritage sites)": 3,
        "Spiritual & religious traveler": 4,
        "Food & culinary explorer": 5,
        "Photography & scenic views seeker": 6,
        "Luxury & relaxation traveler": 7,
        "Business traveler": 8
    },
    "Physical Activity Level": {
        'Very active (hiking, long walks, adventure sports)': 0,
        'Moderately active (walking tours, short hikes)': 1,
        'Less active (prefer easy access locations, relaxation)': 2
    },
    "Experience Level": {
        "First-time traveler": 0,
        "Have traveled occasionally": 1,
        "Frequent traveler": 2
    }
}

# Loop to encode all columns in input_data
    for column in input_data.columns:
        if column in manual_encoding:
            input_data[column] = input_data[column].map(manual_encoding[column])
    print(input_data)
    predicted_category_encoded = model.predict(input_data)
    # Decode the predicted category back to original words
    predicted_category = (lambda x: [
        "Adventure & Unique Experiences",
        "Beaches & Coastal Areas",
        "Historical & Cultural Sites",
        "Nature & Wildlife",
    ][x[0]])(predicted_category_encoded)

    return predicted_category  # Return the decoded category string

if submit:
    st.session_state.recommendation = predict_dt(
        travel_group=travel_group,
        budget=budget,
        accomodation=accomodation,
        activity_interest=activity_interest,
        physical_activity_level=physical_activity_level,
        experience_level=experience_level
    )
    # Trigger a rerun to update the displayed recommendation
    st.rerun()