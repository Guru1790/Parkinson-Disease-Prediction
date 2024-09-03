import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to load and preprocess the dataset
@st.cache_data
def load_data():
    # Load the dataset
    data = pd.read_csv('C:\\Users\\91762\\Desktop\\PK D\\parkinsons.csv')  # Replace with your file path
    return data

# Function to train the model
@st.cache_resource
def train_model():
    data = load_data()
    
    # Feature selection - Dropping the 'name' column as it is not a feature
    X = data.drop(columns=['name', 'status'])
    y = data['status']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train a model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    return model, scaler, X_train, X_test, y_train, y_test

# Load and train the model
model, scaler, X_train, X_test, y_train, y_test = train_model()

# Streamlit application
st.title("Parkinson's Disease Prediction")
st.write("This app predicts whether a person has Parkinson's disease based on their voice measurements.")

# CSS to set background image
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://sl.bing.net/kGNg0cV8C4W");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
'''

# Inject CSS with background image into the app
st.markdown(page_bg_img, unsafe_allow_html=True)

# Create input fields for the features
st.sidebar.header("Input Features")
MDVP_Fo_Hz = st.sidebar.number_input("MDVP:Fo(Hz)", value=119.992)
MDVP_Fhi_Hz = st.sidebar.number_input("MDVP:Fhi(Hz)", value=157.302)
MDVP_Flo_Hz = st.sidebar.number_input("MDVP:Flo(Hz)", value=74.997)
MDVP_Jitter_percent = st.sidebar.number_input("MDVP:Jitter(%)", value=0.00784)
MDVP_Jitter_Abs = st.sidebar.number_input("MDVP:Jitter(Abs)", value=0.00007)
MDVP_RAP = st.sidebar.number_input("MDVP:RAP", value=0.0037)
MDVP_PPQ = st.sidebar.number_input("MDVP:PPQ", value=0.00554)
Jitter_DDP = st.sidebar.number_input("Jitter:DDP", value=0.01109)
MDVP_Shimmer = st.sidebar.number_input("MDVP:Shimmer", value=0.04374)
MDVP_Shimmer_dB = st.sidebar.number_input("MDVP:Shimmer(dB)", value=0.426)
Shimmer_APQ3 = st.sidebar.number_input("Shimmer:APQ3", value=0.02182)
Shimmer_APQ5 = st.sidebar.number_input("Shimmer:APQ5", value=0.0313)
MDVP_APQ = st.sidebar.number_input("MDVP:APQ", value=0.02971)
Shimmer_DDA = st.sidebar.number_input("Shimmer:DDA", value=0.06545)
NHR = st.sidebar.number_input("NHR", value=0.02211)
HNR = st.sidebar.number_input("HNR", value=21.033)
RPDE = st.sidebar.number_input("RPDE", value=0.414783)
DFA = st.sidebar.number_input("DFA", value=0.815285)
spread1 = st.sidebar.number_input("spread1", value=-4.813031)
spread2 = st.sidebar.number_input("spread2", value=0.266482)
D2 = st.sidebar.number_input("D2", value=2.301442)
PPE = st.sidebar.number_input("PPE", value=0.284654)

# Store inputs into a list
input_features = [
    MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percent, MDVP_Jitter_Abs,
    MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3,
    Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2,
    D2, PPE
]

# Predict function
def predict_parkinson(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return "Diseased" if prediction[0] == 1 else "Not Diseased"

# Run the prediction
if st.button("Predict"):
    result = predict_parkinson(input_features)
    st.write(f"Prediction result: **{result}**")

# Display evaluation metrics
if st.checkbox("Show Evaluation Metrics"):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.write("Confusion Matrix:")
    st.text(confusion_matrix(y_test, y_pred))
