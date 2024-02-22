import streamlit as st
import numpy as np
import pickle
from streamlit_option_menu import option_menu
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title= "Singapore Resale Flat Price Prediction App",
                   layout= "wide",
                   initial_sidebar_state= "expanded")

st.title("Singapore Resale Flat Prices Predicting")

st.markdown("""
            <style>
            .stApp {
            background-image: url("https://wallpaper-mania.com/wp-content/uploads/2018/09/High_resolution_wallpaper_background_ID_77700014427.jpg");
            background-size: cover;
            }

            [data-testid="stSidebar"] {
            background-image: url('https://wallpaper-mania.com/wp-content/uploads/2018/09/High_resolution_wallpaper_background_ID_77700459179.jpg');
            background-size: cover;
            }

            </style>""", unsafe_allow_html= True)

with st.sidebar:
    selected = option_menu(None, ["Home" , "Analysis", "Predict Resale Price"],
                            icons = ["house-door-fill"],
                            default_index = 0 ,
                            orientation = "v",
                            styles={"nav-link": {"font-size": "30px", "text-align": "centre", "margin": "0px", 
                                                "--hover-color": "rgb(120, 140, 168)"},
                                   "icon": {"font-size": "30px"},
                                   "container" : {"max-width": "6000px"},
                                   "nav-link-selected": {"background-color":" rgb(172, 168, 140)"}})
    

df = pd.read_csv(r"data.csv")

def storey_range():
    average_prices= df.groupby('storey_range')['resale_price'].mean().reset_index()
    average_prices_sorted = average_prices.sort_values(by='resale_price', ascending=False)
    top_10 = average_prices_sorted.head(10)

    fig = px.bar(top_10, x='storey_range', y='resale_price', title='Top 10 Storey Range listed by the Average Resale Price', labels={'storey_range': 'Storey Range', 'resale_price': 'Resale Price'}, color='storey_range')
    fig.update_traces()
    st.plotly_chart(fig, use_container_width= True)

def town():
    average_prices = df.groupby('town')['resale_price'].mean().reset_index()
    average_prices_sorted = average_prices.sort_values(by='resale_price', ascending=False)
    top_10 = average_prices_sorted.head(10)

    fig = px.bar(top_10, x='town', y='resale_price', title='Top 10 Towns listed by the average resale price', labels={'town': 'Town', 'resale_price': 'Resale Price'}, color = "town")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width= True)

def flat_model():
    average_prices = df.groupby('flat_model')['resale_price'].mean().reset_index()
    average_prices_sorted = average_prices.sort_values(by='resale_price', ascending=False)
    top_10 = average_prices_sorted.head(10)

    fig = px.bar(top_10, x='flat_model', y='resale_price', title='Top 10 Flat Model listed by the Average Resale Price', labels={'flat_model': 'Flat Model', 'resale_price': 'Resale Price'}, color = 'flat_model')
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width= True)

def flat_type():
    average_prices_per_type = df.groupby('flat_type')['resale_price'].mean().reset_index()
    average_prices_per_type_sorted = average_prices_per_type.sort_values(by='resale_price', ascending=False)

    fig = px.bar(average_prices_per_type_sorted, x='flat_type', y='resale_price', title='Average Price Range per Flat Type', labels={'flat_type': 'Flat Type', 'resale_price': 'Resale Price'}, color= 'flat_type')
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width= True)

def pie():

    average_prices_per_type = df.groupby('flat_type')['resale_price'].mean().reset_index()
    hover_data_text = ['Flat Type: {}\nAverage Price: ${:,.2f}'.format(flat_type, avg_price) 
                   for flat_type, avg_price in zip(average_prices_per_type['flat_type'], 
                                                    average_prices_per_type['resale_price'])]
    
    fig = px.pie(average_prices_per_type, values='resale_price', names='flat_type', title='Average Price per Flat Type', 
             hover_data={'flat_type': False, 'resale_price': True}, labels={'resale_price': 'Average Price'},
             height=600, width=800)
    fig.update_traces(textinfo='percent+label', textposition='inside', hole=0.4)
    st.plotly_chart(fig, use_container_width= True)

def resale_price():
    plt.figure(figsize=(10, 6))
    plt.scatter(df['lease_commence_date'], df['resale_price'])
    plt.title('Resale Price over Lease Commencement Date')
    plt.xlabel('Lease Commencement Date')
    plt.ylabel('Resale Price')
    plt.grid(True)

    st.pyplot(plt.gcf())

    

if selected == "Home":
    col1,col2 =st.columns(2, gap = 'medium')

    col1.markdown("### :orange[Title] : Singapore Resale Flat Prices Predicting Using Python Scripting and Streamlit")
    col1.markdown("### :orange[Overview] : This Streamlit app is used to predict the resale price of singapore flats. This process is done by deploying the regression Machine learning model to predict the resale price of the property by using the given data. And Analyse of the features of the given data set")
    col1.markdown("### :orange[Technologies Used] : Python, Streamlit, Pandas, Matplotlib, Plotly Express, Seaborn, Scikit Learn, Pickle and Numpy")

if selected == "Analysis":
    st.write(" ")
    storey_range()
    town()
    st.write(" ")
    flat_model()
    pie()
    flat_type()
    resale_price()


if selected == "Predict Resale Price":

    dict_flat_type = {'3 ROOM': 2, '4 ROOM': 3, '5 ROOM': 4, '2 ROOM': 1, 'EXECUTIVE': 5, '1 ROOM': 0,
                        'MULTI-GENERATION': 6}
     
    flat_type_options = ['3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', '1 ROOM', 'MULTI-GENERATION']

    dict_flat_model = {'Improved': 5, 'New Generation': 12, 'Model A': 8, 'Standard': 17, 'Simplified': 16,
                        'Premium Apartment': 13, 'Maisonette': 7, 'Apartment': 3, 'Model A2': 10,
                        'Type S1': 19, 'Type S2': 20, 'Adjoined flat': 2, 'Terrace': 18, 'DBSS': 4,
                        'Model A-Maisonette': 9, 'Premium Maisonette': 15, 'Multi Generation': 11,
                        'Premium Apartment Loft': 14, 'Improved-Maisonette': 6, '2-room': 0, '3Gen': 1}
     
    flat_model_options = ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
                            'Premium Apartment', 'Maisonette', 'Apartment', 'Model A2',
                            'Type S1', 'Type S2', 'Adjoined flat', 'Terrace', 'DBSS',
                            'Model A-Maisonette', 'Premium Maisonette', 'Multi Generation',
                            'Premium Apartment Loft', 'Improved-Maisonette', '2-room', '3Gen']
    
    dict_town = {'ANG MO KIO': 0, 'BEDOK': 1, 'BISHAN': 2, 'BUKIT BATOK': 3, 'BUKIT MERAH': 4,
       'BUKIT PANJANG': 5, 'BUKIT TIMAH': 6, 'CENTRAL AREA': 7, 'CHOA CHU KANG': 8,
       'CLEMENTI': 9, 'GEYLANG': 10, 'HOUGANG': 11, 'JURONG EAST': 12, 'JURONG WEST': 13,
       'KALLANG/WHAMPOA': 14, 'MARINE PARADE': 15, 'PASIR RIS': 16, 'PUNGGOL': 17,
       'QUEENSTOWN': 18, 'SEMBAWANG': 19, 'SENGKANG': 20, 'SERANGOON': 21, 'TAMPINES': 22,
       'TOA PAYOH': 23, 'WOODLANDS': 24, 'YISHUN': 25}
    

    town_options = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
       'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
       'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
       'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
       'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
       'TOA PAYOH', 'WOODLANDS', 'YISHUN']
    
    story_option = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
    
    with st.form("my_form"):
        col1,col2,col3=st.columns([5,2,5])

        with col1:
            st.write()

            #['floor_area_log',  'remaining_lease_log', 'flat_type_enc', 'flat_model_enc', 'town_enc', 'story_range_log', 'resale_price_log']

            area = st.text_input("Enter the Floor Area (Min:30 & Max:280)")
            lease = st.text_input("Enter the Remaining Lease year (Min:40 to Max: 97)")
            type = st.selectbox("Flat Type", flat_type_options, key = 1)

        with col3:
            st.write()
            flat_model = st.selectbox("Select The Flat Model", flat_model_options, key= 2)
            town = st.selectbox("Sselect The Town", town_options, key = 3)
            story = st.selectbox("Select The Story Range", story_option, key = 4)
            submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")
            st.markdown("""
                <style>
                div.stButton > button:first-child {
                    background-color: #009999;
                    color: white;
                    width: 100%;
                }
                </style>
            """, unsafe_allow_html=True)

        flag=0 
        pattern = "^(?:\d+|\d*\.\d+)$"

        for i in [area, lease]:             
            if re.match(pattern, i):
                pass
            else:                    
                flag=1  
                break     

        if submit_button and flag == 1:   
            if len(i)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",i)

        if submit_button and flag == 0:
            
            with open(r"regression_model_pkl", 'rb') as f:
                model = pickle.load(f)

            with open(r"scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)

            test_data = np.array([[np.log(float(str(area))), np.log(float(str(lease))), dict_flat_type[type], dict_flat_model[flat_model], dict_town[town], np.log(float(str(story)))]])

            # test_data = np.array([[np.log(float(str(area))), np.log(float(str(lease))), float(str(dict[type])), float(str(dict[flat_model])), float(str(dict[town])), np.log(float(str(story)))]])

            test1 = scaler.transform(test_data)

            pred = model.predict(test1)[0]

            st.write('## :green[Predicted selling price:] ', np.exp(pred))
