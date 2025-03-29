import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# -------------------------------
# Load Model and Encoders
# -------------------------------

# Load the trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the LabelEncoders dictionary
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Define the expected feature names (order must match training)
feature_columns = [
    'Color',
    'Year of car manufacture',
    'Length',
    'Width',
    'Brand Name',
    'Transmission type',
    'Seats',
    'Body type',
    'city',
    'Car model',
    'kilometer driven',
    'Insurance Validity_value',
    'Fuel type',
    'Number of previous owners',
    'Engine Displacement_value(cc)'
]

# -------------------------------
# Streamlit App Layout
# -------------------------------

st.title("CAR DHEKO - Used Car Price Prediction")

# Optional: Display a logo or image at the top
image = Image.open(r"E:\GUVI Projects\Cardehko\cardekho-logo_startuptalky.jpg")
st.image(image, use_container_width=True)

st.write(
    "Enter the details of the car below to get an estimated price. "
    "Make sure all inputs are accurate for the best prediction!"
)

# -------------------------------
# Function to Get User Input
# -------------------------------
def get_user_input():
    # Create a grid layout with 3 columns per row
    cols = st.columns(3)
    
    # Row 1
    body_type = cols[0].selectbox("Body type", label_encoders['Body type'].classes_)
    num_owners = cols[1].number_input("Number of previous owners", min_value=0, max_value=10, value=0)
    seats = cols[2].number_input("Seats", min_value=2, max_value=7, value=4)
    
    # Row 2
    city = cols[0].selectbox("City", label_encoders['city'].classes_)
    kilometers_driven = cols[1].number_input("Kilometers driven", min_value=0, max_value=200000, value=5000)
    car_model = cols[2].selectbox("Car model", label_encoders['Car model'].classes_)
    
    # Row 3
    year_of_manufacture = cols[0].number_input("Year of car manufacture", min_value=2000, max_value=2024, value=2015)
    length = cols[1].number_input("Length (mm)", min_value=2000, max_value=6000, value=4000)
    fuel_type = cols[2].selectbox("Fuel type", label_encoders['Fuel type'].classes_)
    
    # Row 4
    width = cols[0].number_input("Width (mm)", min_value=1000, max_value=2500, value=1800)
    color = cols[1].selectbox("Color", label_encoders['Color'].classes_)
    brand_name = cols[2].selectbox("Brand Name", label_encoders['Brand Name'].classes_)
    
    # Row 5
    insurance_validity = cols[0].selectbox("Insurance Validity_value", label_encoders['Insurance Validity_value'].classes_)
    transmission_type = cols[1].selectbox("Transmission type", label_encoders['Transmission type'].classes_)
    engine_displacement = cols[2].number_input("Engine Displacement_value(cc)", min_value=500, max_value=5000, value=1500)
    
    # Assemble input features into a DataFrame in the expected order:
    # Order: Color, Year of car manufacture, Length, Width, Brand Name, Transmission type,
    # Seats, Body type, city, Car model, kilometer driven, Insurance Validity_value,
    # Fuel type, Number of previous owners, Engine Displacement_value(cc)
    features = pd.DataFrame(
        [[
            color,                      # Color
            year_of_manufacture,        # Year of car manufacture
            length,                     # Length (mm)
            width,                      # Width (mm)
            brand_name,                 # Brand Name
            transmission_type,          # Transmission type
            seats,                      # Seats
            body_type,                  # Body type
            city,                       # city
            car_model,                  # Car model
            kilometers_driven,          # kilometer driven
            insurance_validity,         # Insurance Validity_value
            fuel_type,                  # Fuel type
            num_owners,                 # Number of previous owners
            engine_displacement         # Engine Displacement_value(cc)
        ]],
        columns=feature_columns
    )
    return features

# Get user input features
user_features = get_user_input()

# Display the user input
st.write("### User Input Features:")
st.write(user_features)

# Prediction button
if st.button('Predict Price'):
    with st.spinner("Predicting..."):
        try:
            # Encode the categorical features using label encoders
            for col in ['Color', 'Transmission type', 'Body type', 'Brand Name', 
                        'Fuel type', 'city', 'Car model', 'Insurance Validity_value']:
                user_features[col] = label_encoders[col].transform(user_features[col])
            
            # Print the final columns for debugging (optional)
            st.write("Final feature columns:", user_features.columns.tolist())
            
            # Make the prediction using the loaded model
            prediction = model.predict(user_features)
            
            # Display the predicted price
            st.success(f'Estimated Car Price: ₹ {prediction[0]:,.2f}')
            
        except ValueError as ve:
            st.error(f"Value error: {ve}. Please check your input.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
