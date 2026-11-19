# Car Price Prediction — Regression + Web Deployment

## Files
- `cars_data.csv`: dataset
- `Car_Price_Regression_Models.ipynb`: full ML workflow
- `app.py`: Streamlit web application
- `car_price_model.joblib`: trained model and preprocessing objects
- `requirements.txt`: dependencies

## Run locally
1. Open a terminal in this folder.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Start the web app:
   `streamlit run app.py`
4. Open the local URL printed in the terminal (usually http://localhost:8501).

The web app asks for car details and displays an estimated price. The saved model was trained on the provided dataset. Predictions are estimates and may not reflect current market prices.
