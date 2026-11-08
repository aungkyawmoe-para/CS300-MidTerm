import streamlit as st
import pandas as pd
import pickle

df = pd.read_csv('rice_price_dataset_dirty.csv')

df.columns = df.columns.str.strip()
st.header("Rice Price Prediction App")
result = 0
def load_modelBestR2():
    with open('ricepredictBestR2.pkl','rb') as f:
        return pickle.load(f)


with st.sidebar:
    st.image('parami.jpg')
    st.text("Student Name: Aung Kyaw Moe")
    st.text("Class Name: Introduction to ML")

    
with st.form('Rice Data'):
    col1,col2=st.columns(2)
    with col1:

        var1 = st.selectbox('Select which region your rice came from:', ['North', 'West', 'East', 'South'])
        var2 = st.selectbox('Select if your rice type:', ['Jasmine', 'Brown', 'Basmati', 'White'])
        var7 = st.selectbox('Select if you rice export or import:',['Import', 'Export'])

        var3 = st.number_input('Select rice demand in tons',
                        min_value=0.0,value= 152.8
                        )

    with col2:
        var4 = st.number_input('Enter rice supply in tons',
                            min_value=0.0,value= 132.0
                            ) 
        var5=st.slider("Choose rainfall measured in mm: ",
                    max_value=500.0,
                    min_value=0.0,step=0.5)
        var6=st.slider("Enter storage temperature: ",
                    max_value=46.0,
                    min_value=10.0,step=0.1)


    predict_button = st.form_submit_button('Predict Rice Price')

if predict_button:
    input_data=pd.DataFrame(
            [
                {
                    'Region':var1, 
                    'Rice_Type':var2, 
                    'Demand_Tons':var3, 
                    'Supply_Tons':var4,
                    'Rainfall_mm':var5, 
                    'Temperature_C':var6, 
                    'Import_Export':var7
                }
            ]
        )
    model=load_modelBestR2()
    result=model.predict(input_data)[0]
    st.write(f"Rice price per kilogram: {result:.2f} $")


with st.expander("Dataset"):
    st.table(df)

with st.expander('Dataset and Model overview'):
    st.markdown("Datatset Overview")
    st.markdown("""
    - Dataset include 1020 rows, and 10 columns
    - Column names: 'Rice_ID', 'Region', 'Rice_Type', 'Price_per_kg', 'Demand_Tons',
    'Supply_Tons', 'Rainfall_mm', 'Temperature_C', 'Import_Export', 'Date'
    - Missing values present in all columns

    """)
    st.markdown("Model Overview")
    st.markdown("""
    Model Evaluation:
    - MAE:14.45
    - MSE:634.53
    - RMSE:25.19
    - R2:0.731 or 73.1%
    - KNeighbor Regressor with n=5 is used for the model paired with standard scaler, and one hot encoder for preprocessing.
    - Data cleaning step: Filling missng data, mode is used for categorical columns and mean is used numerical columns excluding temperature and rainfall which used bfill ffill.
        
    """)


