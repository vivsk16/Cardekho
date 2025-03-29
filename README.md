#Car Dheko – Used Car Price Prediction
Overview
This project predicts used car prices based on historical data from Car Dheko. It includes data analysis, model training, and a Streamlit app for real-time price predictions.

Files
Car_Dheko.ipynb: Main notebook summarizing the project steps.

EDA.ipynb: Notebook for exploratory data analysis.

Cars_ML.ipynb: Notebook for machine learning model development and evaluation.

App.py: Streamlit app that loads the trained model and label encoders to predict car prices.

best_model.pkl: Pickle file of the best-performing model.

final data.csv: Cleaned dataset used for model training.

label_encoders.pkl: Pickle file containing label encoders for categorical features.

all_car_details.csv: Combined raw or intermediate dataset (if needed).

How to Run
Install Dependencies (e.g., pandas, scikit-learn, streamlit).

Open and Explore the notebooks (Car_Dheko.ipynb, EDA.ipynb, Cars_ML.ipynb) for data cleaning, EDA, and model training.

Run the Streamlit App:

bash
Copy
Edit
streamlit run App.py
This will start the app in your web browser, allowing you to input car details and get price predictions.

License
This project is released under the MIT License (or another license of your choice).
