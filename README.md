# **My Body Weight Project**
A very interesting time series prediction project that used deep learning LSTM model using my own body weight data from the past 5 years.
## Project Summary
As a person who was obese and lost 100lbs, I am very sensitive about my own body weight. Using my own body weight data for the past 5+ years, I trained a LSTM model that has an testing RMSE of 0.5kg which can predict my next day body weight using 7 days of previous body weight data, which I am happy with, it is pretty accurate for me. I also deployed the project using streamlit and made an webpage application that displays a interactive visualization and also has a built in body weight prediction function.
## Tools Used
Python, Jupyter Notebook, Streamlit
## Data Science Techniques Used
Plotly, Streamlit, Pandas, Numpy, PyTorch, EDA, Machine Learning, Deep Learning, LSTM, Matplotlib, etc.
## Files
- Project Code.ipynb (Jupyter notebook file that contains all the codes for the project along with detailed markdown cells explaining everything you need to know)
- My Body Weight.csv (raw data of my own 5 year body weight data)
- app.py (the main streamlit python file)
- lstm.py (codes that stored the function to create the LSTM model, used in app.py)
- model_lstm.pth (best trained LSTM model)
- weight_scaler.pkl (the saved scalar in a file to denormalize)
## Usage
To view the entire project, check out the jupyter notebook file (Project Code.ipynb), details are all explained with markdown cells. To run the demo application I created, click this link: [Eric Mei Body Weight Project](https://ericmeibodyweight.streamlit.app).
## Data Source
The raw data used in this project was my own body weight recordings over the past 5+ years.