# 🛠 Import required packages
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from sklearn.preprocessing import LabelEncoder

# 🔧 Set Streamlit config (MUST be first Streamlit command)

st.set_page_config(page_title="🛫 Flight Price Predictor", layout="centered")


# 🎨 Set background image using local file in same folder
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 🖼 Add background from current folder
add_bg_from_local("C:\\Users\\induu\\Downloads\\flight_Project\\flight_dream.jpg")

# 📦 Load trained model (in same folder)
with open("C:\\Users\\induu\\Downloads\\flight.pkl", "rb") as f:
    model = joblib.load(f)



# 🧾 App UI with emojis
st.title("🛩 Flight Price Predictor")
st.markdown("---")


airline = st.selectbox("🛫 Airline", ['Indigo', 'Air_India', 'Vistara', 'SpiceJet', 'GO_FIRST', 'AirAsia'])
source_city = st.selectbox('🌆 Source City', ['Delhi', 'Mumbai', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata'])
departure_time = st.selectbox('⏰ Departure Time', ['Morning', 'Early Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'])
stops = st.selectbox("🛑 Total Stops", [0, 1, 2, 3])
arrival_time = st.selectbox('🕓 Arrival Time', ['Morning', 'Early Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'])
destination_city = st.selectbox('🏙 Destination City', ['Mumbai', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Delhi'])
class_type = st.selectbox('💺 Travel Class', ['Business', 'Economy'])
duration_minutes = st.slider("🕒 Flight Duration (Hours)", 1, 50, 1)
days_left = st.slider('📅 Days Left Until Departure', 1, 50, 10)

# 🧠 Predict button
if st.button("📍 Predict Flight Price"):
    df = pd.DataFrame({
       
        'airline': [airline],
        'source_city': [source_city],
        'departure_time': [departure_time],
        'stops': [stops],
        'arrival_time': [arrival_time],
        'destination_city': [destination_city],
        'class': [class_type],
        'duration_minutes': [duration_minutes],
        'days_left': [days_left]
    })

    # 🔄 Label Encoding
    le = LabelEncoder()
    categorical_cols = ['airline', 'source_city', 'departure_time', 'arrival_time', 'destination_city', 'class']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # ✅ Prepare input
    input_data = np.array([[
    
        df['airline'][0],
        df['source_city'][0],
        df['departure_time'][0],
        df['stops'][0],
        df['arrival_time'][0],
        df['destination_city'][0],
        df['class'][0],
        df['duration_minutes'][0],
        df['days_left'][0]
    ]], dtype=int)

    # 🎯 Predict and show result
    prediction = np.exp(model.predict(input_data))
    st.success(f"🧾 Estimated Flight Price: ₹ {int(round(prediction[0],2))}")