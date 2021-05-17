import streamlit as st
import pandas as pd
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import base64
import seaborn as sns
import datetime as dt
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import StandardScaler
import urllib
import plotly.express as px
import plotly.graph_objects as go
import calendar

st.set_page_config(
    page_title = 'Samer Haidar Streamlit App',
    page_icon = '📊',
    layout = 'wide')

@st.cache(allow_output_mutation=True)
def read_file(file):
    df = pd.read_csv(file)
    return df

file = 'Data.csv'
data = read_file(file)

image = Image.open("data-original.jpg")
st.sidebar.image(image)
st.text('')

purpose = st.sidebar.selectbox("Choose an Option", ["Business KPIs", "Profitability Analysis", "Sales Forecasting", 'RFM Analysis and Segmentation'])


if purpose == "Business KPIs":
        
        date = st.sidebar.date_input("Date Range", [dt.date(2011, 5, 18), dt.date(2014, 12, 31)])

        data['Order Date'] = pd.to_datetime(data['Order Date'])

        start_date = pd.to_datetime(date[0])
        end_date = pd.to_datetime(date[1])

        after_start_date = data["Order Date"] >= start_date
        before_end_date = data["Order Date"] <= end_date

        between_two_dates = after_start_date & before_end_date
        filtered_dates = data.loc[between_two_dates]

        categories = data['Category'].unique()
        all_cat = np.array(['All'])
        categories = np.concatenate((all_cat, categories))

        purpose1 = st.sidebar.radio('Choose a Category', categories)    

        if purpose1 == 'All':


            kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

            with kpi1:
                title1 = 'Sales'
                st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                
                number1 = round(filtered_dates['Sales'].sum())
                st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
        

            with kpi2:
                title2 = 'Profit'
                st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                
                number2 = round(filtered_dates['Profit'].sum())
                st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

            with kpi3:
                title3 = 'Quantity'
                st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                
                number3 = round(filtered_dates['Quantity'].sum())
                st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

            with kpi4:
                title4 = 'Customers'
                st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                
                number4 = round(filtered_dates['Customer ID'].nunique())
                st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

            plot1, plot2 = st.beta_columns(2)

            with plot1:

                st.text('')
                st.text('')
                st.text('')

                sales_per_day = data.groupby(by = 'Order Date')['Sales'].sum()
                sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                
                fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                fig.update_layout(title_text= 'Sales over Specified Date Range',
                    title_font_size=22)
                

                st.plotly_chart(fig)

            with plot2:

                st.text('')
                st.text('')
                st.text('')

                profit_per_day = data.groupby(by = 'Order Date')['Profit'].sum()
                profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                
                fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                fig1.update_layout(title_text= 'Profit over Specified Date Range',
                    title_font_size=22)

                st.plotly_chart(fig1)

            with plot1:

                quantity_per_day = data.groupby(by = 'Order Date')['Quantity'].sum()
                quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                
                fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                    title_font_size=22)

                st.plotly_chart(fig2)

            with plot2:


                customers_per_day = data.groupby(by = 'Order Date')['Customer ID'].nunique()
                customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                
                fig1 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                fig1.update_layout(title_text= 'Customers over Specified Date Range',
                    title_font_size=22)

                st.plotly_chart(fig1)


        if purpose1 == 'Technology':


            item_tech = data[data['Category'] == 'Technology']['Sub-Category'].unique()
            all_tech = np.array(['All'])
            item_tech = np.concatenate((all_tech, item_tech))

            purpose_tech = st.sidebar.radio('Choose an Item', item_tech)

            if purpose_tech == 'All':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Category'] == 'Technology']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Category'] == 'Technology']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Category'] == 'Technology']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Category'] == 'Technology']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Category'] == 'Technology'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Category'] == 'Technology'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Category'] == 'Technology'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Category'] == 'Technology'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)


            if purpose_tech == 'Accessories':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Accessories']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Accessories']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Accessories']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Accessories']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Sub-Category'] == 'Accessories'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Sub-Category'] == 'Accessories'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Sub-Category'] == 'Accessories'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Sub-Category'] == 'Accessories'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

            if purpose_tech == 'Phones':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Phones']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Phones']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Phones']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Phones']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Sub-Category'] == 'Phones'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Sub-Category'] == 'Phones'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Sub-Category'] == 'Phones'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Sub-Category'] == 'Phones'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)


            if purpose_tech == 'Copiers':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Copiers']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Copiers']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Copiers']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Copiers']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Sub-Category'] == 'Copiers'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Sub-Category'] == 'Copiers'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Sub-Category'] == 'Copiers'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Sub-Category'] == 'Copiers'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)


            if purpose_tech == 'Machines':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Machines']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Machines']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Machines']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Machines']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Sub-Category'] == 'Machines'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Sub-Category'] == 'Machines'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Sub-Category'] == 'Machines'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Sub-Category'] == 'Machines'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)


        if purpose1 == 'Furniture':
            

            item_furn = data[data['Category'] == 'Furniture']['Sub-Category'].unique()

            all_furn = np.array(['All'])
            item_furn = np.concatenate((all_furn, item_furn))

            purpose_furn = st.sidebar.radio('Choose an Item', item_furn)

            if purpose_furn == 'All':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Category'] == 'Furniture']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Category'] == 'Furniture']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Category'] == 'Furniture']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Category'] == 'Furniture']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Category'] == 'Furniture'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Category'] == 'Furniture'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Category'] == 'Furniture'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Category'] == 'Furniture'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

            if purpose_furn == 'Chairs':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Chairs']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Chairs']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Chairs']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Chairs']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Sub-Category'] == 'Chairs'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Sub-Category'] == 'Chairs'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Sub-Category'] == 'Chairs'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Sub-Category'] == 'Chairs'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)


            if purpose_furn == 'Tables':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Tables']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Tables']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Tables']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Tables']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Sub-Category'] == 'Tables'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Sub-Category'] == 'Tables'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Sub-Category'] == 'Tables'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Sub-Category'] == 'Tables'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)


            if purpose_furn == 'Bookcases':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Bookcases']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Bookcases']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Bookcases']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Bookcases']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Sub-Category'] == 'Bookcases'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Sub-Category'] == 'Bookcases'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Sub-Category'] == 'Bookcases'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Sub-Category'] == 'Bookcases'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)


            if purpose_furn == 'Furnishings':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Furnishings']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Furnishings']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Furnishings']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Furnishings']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Sub-Category'] == 'Furnishings'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Sub-Category'] == 'Furnishings'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Sub-Category'] == 'Furnishings'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Sub-Category'] == 'Furnishings'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

        
        if purpose1 == 'Office Supplies':
            

            item_off = data[data['Category'] == 'Office Supplies']['Sub-Category'].unique()

            all_off = np.array(['All'])
            item_off = np.concatenate((all_off, item_off))

            purpose_off = st.sidebar.radio('Choose an Item', item_off)

            if purpose_off == 'All':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Category'] == 'Office Supplies']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Category'] == 'Office Supplies']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Category'] == 'Office Supplies']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Category'] == 'Office Supplies']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Category'] == 'Office Supplies'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Category'] == 'Office Supplies'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Category'] == 'Office Supplies'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Category'] == 'Office Supplies'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

            if purpose_off == 'Blinders':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Blinders']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Blinders']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Blinders']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Blinders']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Sub-Category'] == 'Blinders'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Sub-Category'] == 'Blinders'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Sub-Category'] == 'Blinders'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Sub-Category'] == 'Blinders'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)



            if purpose_off == 'Supplies':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Supplies']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Supplies']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Supplies']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Supplies']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Sub-Category'] == 'Supplies'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Sub-Category'] == 'Supplies'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Sub-Category'] == 'Supplies'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Sub-Category'] == 'Supplies'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)


            if purpose_off == 'Appliances':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Appliances']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Appliances']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Appliances']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Appliances']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Sub-Category'] == 'Appliances'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Sub-Category'] == 'Appliances'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Sub-Category'] == 'Appliances'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Sub-Category'] == 'Appliances'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)


            if purpose_off == 'Storage':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Storage']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Storage']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Storage']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Storage']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Sub-Category'] == 'Storage'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Sub-Category'] == 'Storage'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Sub-Category'] == 'Storage'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Sub-Category'] == 'Storage'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)



            if purpose_off == 'Art':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Art']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Art']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Art']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Art']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Sub-Category'] == 'Art'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Sub-Category'] == 'Art'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Sub-Category'] == 'Art'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Sub-Category'] == 'Art'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

            if purpose_off == 'Paper':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Paper']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Paper']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Paper']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Paper']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Sub-Category'] == 'Paper'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Sub-Category'] == 'Paper'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Sub-Category'] == 'Paper'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Sub-Category'] == 'Paper'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

            if purpose_off == 'Envelopes':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Envelopes']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Envelopes']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Envelopes']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Envelopes']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Sub-Category'] == 'Envelopes'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Sub-Category'] == 'Envelopes'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Sub-Category'] == 'Envelopes'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Sub-Category'] == 'Envelopes'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

            if purpose_off == 'Fasteners':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Fasteners']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Fasteners']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Fasteners']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Fasteners']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Sub-Category'] == 'Fasteners'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Sub-Category'] == 'Fasteners'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Sub-Category'] == 'Fasteners'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Sub-Category'] == 'Fasteners'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)


            if purpose_off == 'Labels':

                kpi1, kpi2, kpi3, kpi4 = st.beta_columns(4)

                with kpi1:
                    title1 = 'Sales'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title1}</h1>", unsafe_allow_html=True)
                    
                    number1 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Labels']['Sales'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)
            

                with kpi2:
                    title2 = 'Profit'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title2}</h1>", unsafe_allow_html=True)
                    
                    number2 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Labels']['Profit'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

                with kpi3:
                    title3 = 'Quantity'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title3}</h1>", unsafe_allow_html=True)
                    
                    number3 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Labels']['Quantity'].sum())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

                with kpi4:
                    title4 = 'Customers'
                    st.markdown(f"<h1 style='text-align: center; color: DarkBlue;'>{title4}</h1>", unsafe_allow_html=True)
                    
                    number4 = round(filtered_dates[filtered_dates['Sub-Category'] == 'Labels']['Customer ID'].nunique())
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{number4}</h1>", unsafe_allow_html=True)

                plot1, plot2 = st.beta_columns(2)

                with plot1:

                    st.text('')
                    st.text('')
                    st.text('')

                    sales_per_day = data[data['Sub-Category'] == 'Labels'].groupby(by = 'Order Date')['Sales'].sum()
                    sales_per_day = pd.DataFrame(sales_per_day).reset_index(level = 0)
                    
                    fig = px.line(sales_per_day, x='Order Date', y='Sales', range_x=[start_date,end_date], width=700, height= 500)

                    fig.update_layout(title_text= 'Sales over Specified Date Range',
                        title_font_size=22)
                    

                    st.plotly_chart(fig)

                with plot2:
                    st.text('')
                    st.text('')
                    st.text('')
                    profit_per_day = data[data['Sub-Category'] == 'Labels'].groupby(by = 'Order Date')['Profit'].sum()
                    profit_per_day = pd.DataFrame(profit_per_day).reset_index(level = 0)
                    
                    fig1 = px.line(profit_per_day, x='Order Date', y='Profit', range_x=[start_date,end_date], width=700, height= 500)

                    fig1.update_layout(title_text= 'Profit over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig1)

                with plot1:

                    quantity_per_day = data[data['Sub-Category'] == 'Labels'].groupby(by = 'Order Date')['Quantity'].sum()
                    quantity_per_day = pd.DataFrame(quantity_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(quantity_per_day, x='Order Date', y='Quantity', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Quanity Bought over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)

                with plot2:

                    customers_per_day = data[data['Sub-Category'] == 'Labels'].groupby(by = 'Order Date')['Customer ID'].nunique()
                    customers_per_day = pd.DataFrame(customers_per_day).reset_index(level = 0)
                    
                    fig2 = px.line(customers_per_day, x='Order Date', y='Customer ID', range_x=[start_date,end_date], width=700, height= 500)

                    fig2.update_layout(title_text= 'Customers over Specified Date Range',
                        title_font_size=22)

                    st.plotly_chart(fig2)


elif purpose == "Profitability Analysis":



        purpose1 = st.sidebar.radio('Choose an Option', ['Profits by Sub-Category Table', 'Most Profitable Sub-Categories', 'Most Profitable Customers', 'Sales by Locations'])

        if purpose1 == 'Profits by Sub-Category Table': 

            st.title("Sub-Categories' Profits by Year and Month" )
            st.text('')

            sub_category_year = pd.pivot_table(data=data,values='Profit',index=['Sub-Category','year'],columns='month').rename(
                columns=lambda x:calendar.month_name[x])

            st.table(sub_category_year)

        if purpose1 == 'Most Profitable Sub-Categories':
            date = st.sidebar.date_input("Date Range", [dt.date(2011, 5, 14), dt.date(2014, 12, 31)])

            data['Order Date'] = pd.to_datetime(data['Order Date'])

            start_date = pd.to_datetime(date[0])
            end_date = pd.to_datetime(date[1])

            after_start_date = data["Order Date"] >= start_date
            before_end_date = data["Order Date"] <= end_date

            between_two_dates = after_start_date & before_end_date
            filtered_dates = data.loc[between_two_dates]
            
           
                
            fig = px.bar(filtered_dates, x=filtered_dates["Sub-Category"].unique(), 
            y= filtered_dates.groupby('Sub-Category')["Profit"].sum(), width=1000, height= 500,
            labels={
                    "x": "Sub-Category",
                    "y": "Profits"},)

            fig.update_layout( xaxis={'categoryorder':'total descending'}, title_text= 'Profits by Sub-Category', title_font_size=22)
            st.plotly_chart(fig)

        
            
            fig = px.bar(filtered_dates, x=filtered_dates["Sub-Category"].unique(), 
            y= filtered_dates.groupby('Sub-Category')["Sales"].sum(), width=1000, height= 500,
            labels={
                    "x": "Sub-Category",
                    "y": "Sales"},)

            fig.update_layout( xaxis={'categoryorder':'total descending'}, title_text= 'Sales by Sub-Category', title_font_size=22)
            st.plotly_chart(fig)

        
            
            fig = px.bar(filtered_dates, x=filtered_dates["Sub-Category"].unique(), 
            y= filtered_dates.groupby('Sub-Category')["Quantity"].sum(), width=1000, height= 500,
            labels={
                    "x": "Sub-Category",
                    "y": "Quantity"},)

            fig.update_layout( xaxis={'categoryorder':'total descending'}, title_text= 'Quantity Sold by Sub-Category', title_font_size=22)
            st.plotly_chart(fig)

        if purpose1 == 'Most Profitable Customers': 
            date = st.sidebar.date_input("Date Range", [dt.date(2011, 5, 14), dt.date(2014, 12, 31)])

            data['Order Date'] = pd.to_datetime(data['Order Date'])

            start_date = pd.to_datetime(date[0])
            end_date = pd.to_datetime(date[1])

            after_start_date = data["Order Date"] >= start_date
            before_end_date = data["Order Date"] <= end_date

            between_two_dates = after_start_date & before_end_date
            filtered_dates = data.loc[between_two_dates]

            customers = data['Segment'].unique()
            purpose2 = st.sidebar.radio('Choose a Customer Type', customers)

            if purpose2 == 'Consumer':

                # st.header('Customers')
                cust1, cust2, cust3, cust4 = st.beta_columns(4)

                with cust1:
                    title = "Total Customers"
                    st.markdown(f"<h2 style='text-align: center; color: DarkBlue;'>{title}</h2>", unsafe_allow_html=True)

                    number_of_customers = filtered_dates[filtered_dates['Segment'] == 'Consumer']['Customer ID'].nunique() #number of unique customers
                    st.markdown(f"<h2 style='text-align: center; color: red;'>{number_of_customers}</h2>", unsafe_allow_html=True)

                with cust2:

                    before_selected_date = data[data["Order Date"] < start_date]

                    title1 = "New Customers"
                    st.markdown(f"<h2 style='text-align: center; color: DarkBlue;'>{title1}</h2>", unsafe_allow_html=True)

                    new_customers = filtered_dates[(filtered_dates['Segment'] == 'Consumer') & (filtered_dates['Customer ID'].isin(before_selected_date['Customer ID'])== False)]['Customer ID'].nunique() 
                    st.markdown(f"<h2 style='text-align: center; color: red;'>{new_customers}</h2>", unsafe_allow_html=True)

                with cust3:
                    title2 = "Items Bought"
                    st.markdown(f"<h2 style='text-align: center; color: DarkBlue;'>{title2}</h2>", unsafe_allow_html=True)

                    number_of_items = filtered_dates[filtered_dates['Segment'] == 'Consumer']['Quantity'].sum() 
                    st.markdown(f"<h2 style='text-align: center; color: red;'>{number_of_items}</h2>", unsafe_allow_html=True)

                with cust4:
                    title3 = "Profit"
                    st.markdown(f"<h2 style='text-align: center; color: DarkBlue;'>{title3}</h2>", unsafe_allow_html=True)

                    profit = round(filtered_dates[filtered_dates['Segment'] == 'Consumer']['Profit'].sum()) 
                    st.markdown(f"<h2 style='text-align: center; color: red;'>{profit}</h2>", unsafe_allow_html=True)

                k = st.sidebar.slider("How Many Top Customer?",20, 1, 20)
                i = st.sidebar.slider("How Many Top Products?",20, 1, 20)

                cust7, cust8= st.beta_columns(2)

                with cust7:
                    st.text('')
                    st.text('')
                    st.text('')

                    # title3 = "Top "+str(k)+" Profitable Customers"
                    # st.markdown(f"<h2 style='text-align: left; color: DarkBlue;'>{title3}</h2>", unsafe_allow_html=True)

                    top_customers_profits =round(filtered_dates[filtered_dates['Segment'] == 'Consumer'].groupby(by=["Customer Name"])["Profit"].sum()) #revenue of top k customers
                    top_customers_profits = pd.DataFrame(top_customers_profits)
                    top_customers_profits = top_customers_profits.sort_values(by=['Profit'], ascending=False)[0:k]
                    top_customers_profits.reset_index(level=0, inplace=True)

                    fig = go.Figure(data=[go.Table(
                            header=dict(values=['Customer', 'Profit'],
                            #line_color='darkslategray',
                            align='center',font=dict(color='black', size=16)),
                            cells=dict(values=[top_customers_profits['Customer Name'], # 1st column
                            top_customers_profits['Profit']], # 2nd column
                            align='center',font=dict(color='black', size=13)))])
                    
                    fig.update_layout(width=600, height=650, grid_yside="left", title_text= "Top "+str(k)+" Profitable Consumer Customers", title_font_size=22)

                    
                    st.plotly_chart(fig)

                with cust8:
                    st.text('')
                    st.text('')
                    st.text('')

                    # title3 = "Top "+str(k)+" Profitable Customers"
                    # st.markdown(f"<h2 style='text-align: left; color: DarkBlue;'>{title3}</h2>", unsafe_allow_html=True)

                    top_customers_orders =round(filtered_dates[filtered_dates['Segment'] == 'Consumer'].groupby(by=["Customer Name"])["Order ID"].count()) #revenue of top k customers
                    top_customers_orders = pd.DataFrame(top_customers_orders)
                    top_customers_orders = top_customers_orders.sort_values(by=['Order ID'], ascending=False)[0:k]
                    top_customers_orders.reset_index(level=0, inplace=True)

                    fig = go.Figure(data=[go.Table(
                            header=dict(values=['Customer', 'Orders'],
                            #line_color='darkslategray',
                            align='center',font=dict(color='black', size=16)),
                            cells=dict(values=[top_customers_orders['Customer Name'], # 1st column
                            top_customers_orders['Order ID']], # 2nd column
                            align='center',font=dict(color='black', size=13)))])
                    
                    fig.update_layout(width=600, height=650, grid_yside="left", title_text= "Top "+str(k)+" Consumer Customers by Orders", title_font_size=22)

                    
                    st.plotly_chart(fig)

                with cust7:

                    # title3 = "Top "+str(k)+" Profitable Customers"
                    # st.markdown(f"<h2 style='text-align: left; color: DarkBlue;'>{title3}</h2>", unsafe_allow_html=True)

                    top_customers_sub_cat =round(filtered_dates[filtered_dates['Segment'] == 'Consumer'].groupby(by=["Sub-Category"])["Quantity"].sum()) #revenue of top k customers
                    top_customers_sub_cat = pd.DataFrame(top_customers_sub_cat)
                    top_customers_sub_cat = top_customers_sub_cat.sort_values(by=['Quantity'], ascending=False)[0:i]
                    top_customers_sub_cat.reset_index(level=0, inplace=True)

                    fig = go.Figure(data=[go.Table(
                            header=dict(values=['Customer', 'Quantity'],
                            #line_color='darkslategray',
                            align='center',font=dict(color='black', size=16)),
                            cells=dict(values=[top_customers_sub_cat['Sub-Category'], # 1st column
                            top_customers_sub_cat['Quantity']], # 2nd column
                            align='center',font=dict(color='black', size=13)))])
                    
                    fig.update_layout(width=600, height=650, grid_yside="left", title_text= "Top "+str(i)+" Bought Sub-Categories by Consumers", title_font_size=22)

                    
                    st.plotly_chart(fig)

                with cust8:

                    # title3 = "Top "+str(k)+" Profitable Customers"
                    # st.markdown(f"<h2 style='text-align: left; color: DarkBlue;'>{title3}</h2>", unsafe_allow_html=True)

                    top_customers_items =round(filtered_dates[filtered_dates['Segment'] == 'Consumer'].groupby(by=["Product Name"])["Quantity"].count()) #revenue of top k customers
                    top_customers_items = pd.DataFrame(top_customers_items)
                    top_customers_items = top_customers_items.sort_values(by=['Quantity'], ascending=False)[0:i]
                    top_customers_items.reset_index(level=0, inplace=True)

                    fig = go.Figure(data=[go.Table(
                            header=dict(values=['Product Name', 'Quantity'],
                            #line_color='darkslategray',
                            align='center',font=dict(color='black', size=16)),
                            cells=dict(values=[top_customers_items['Product Name'], # 1st column
                            top_customers_items['Quantity']], # 2nd column
                            align='center',font=dict(color='black', size=13)))])
                    
                    fig.update_layout(width=600, height=650, grid_yside="left", title_text= "Top "+str(i)+" Bought Products by Consumer Customers", title_font_size=22)

                    
                    st.plotly_chart(fig)


            if purpose2 == 'Corporate':

                k = st.sidebar.slider("How Many Top Customer?",20, 1, 20)
                i = st.sidebar.slider("How Many Top Products?",20, 1, 20)

                # st.header('Customers')
                cust1, cust2, cust3, cust4 = st.beta_columns(4)

                with cust1:
                    title = "Total Customers"
                    st.markdown(f"<h2 style='text-align: center; color: DarkBlue;'>{title}</h2>", unsafe_allow_html=True)

                    number_of_customers = filtered_dates[filtered_dates['Segment'] == 'Corporate']['Customer ID'].nunique() #number of unique customers
                    st.markdown(f"<h2 style='text-align: center; color: red;'>{number_of_customers}</h2>", unsafe_allow_html=True)

                with cust2:

                    before_selected_date = data[data["Order Date"] < start_date]

                    title1 = "New Customers"
                    st.markdown(f"<h2 style='text-align: center; color: DarkBlue;'>{title1}</h2>", unsafe_allow_html=True)

                    new_customers = filtered_dates[(filtered_dates['Segment'] == 'Corporate') & (filtered_dates['Customer ID'].isin(before_selected_date['Customer ID'])== False)]['Customer ID'].nunique() 
                    st.markdown(f"<h2 style='text-align: center; color: red;'>{new_customers}</h2>", unsafe_allow_html=True)

                with cust3:
                    title2 = "Items Bought"
                    st.markdown(f"<h2 style='text-align: center; color: DarkBlue;'>{title2}</h2>", unsafe_allow_html=True)

                    number_of_items = filtered_dates[filtered_dates['Segment'] == 'Corporate']['Quantity'].sum() 
                    st.markdown(f"<h2 style='text-align: center; color: red;'>{number_of_items}</h2>", unsafe_allow_html=True)

                with cust4:
                    title3 = "Profit"
                    st.markdown(f"<h2 style='text-align: center; color: DarkBlue;'>{title3}</h2>", unsafe_allow_html=True)

                    profit = round(filtered_dates[filtered_dates['Segment'] == 'Corporate']['Profit'].sum()) 
                    st.markdown(f"<h2 style='text-align: center; color: red;'>{profit}</h2>", unsafe_allow_html=True)


                cust7, cust8= st.beta_columns(2)

                with cust7:
                    st.text('')
                    st.text('')
                    st.text('')

                    # title3 = "Top "+str(k)+" Profitable Customers"
                    # st.markdown(f"<h2 style='text-align: left; color: DarkBlue;'>{title3}</h2>", unsafe_allow_html=True)

                    top_customers_profits =round(filtered_dates[filtered_dates['Segment'] == 'Corporate'].groupby(by=["Customer Name"])["Profit"].sum()) #revenue of top k customers
                    top_customers_profits = pd.DataFrame(top_customers_profits)
                    top_customers_profits = top_customers_profits.sort_values(by=['Profit'], ascending=False)[0:k]
                    top_customers_profits.reset_index(level=0, inplace=True)

                    fig = go.Figure(data=[go.Table(
                            header=dict(values=['Customer', 'Profit'],
                            #line_color='darkslategray',
                            align='center',font=dict(color='black', size=16)),
                            cells=dict(values=[top_customers_profits['Customer Name'], # 1st column
                            top_customers_profits['Profit']], # 2nd column
                            align='center',font=dict(color='black', size=13)))])
                    
                    fig.update_layout(width=600, height=650, grid_yside="left", title_text= "Top "+str(k)+" Profitable Corporate Customers", title_font_size=22)

                    
                    st.plotly_chart(fig)

                with cust8:
                    st.text('')
                    st.text('')
                    st.text('')

                    # title3 = "Top "+str(k)+" Profitable Customers"
                    # st.markdown(f"<h2 style='text-align: left; color: DarkBlue;'>{title3}</h2>", unsafe_allow_html=True)

                    top_customers_orders =round(filtered_dates[filtered_dates['Segment'] == 'Corporate'].groupby(by=["Customer Name"])["Order ID"].count()) #revenue of top k customers
                    top_customers_orders = pd.DataFrame(top_customers_orders)
                    top_customers_orders = top_customers_orders.sort_values(by=['Order ID'], ascending=False)[0:k]
                    top_customers_orders.reset_index(level=0, inplace=True)

                    fig = go.Figure(data=[go.Table(
                            header=dict(values=['Customer', 'Orders'],
                            #line_color='darkslategray',
                            align='center',font=dict(color='black', size=16)),
                            cells=dict(values=[top_customers_orders['Customer Name'], # 1st column
                            top_customers_orders['Order ID']], # 2nd column
                            align='center',font=dict(color='black', size=13)))])
                    
                    fig.update_layout(width=600, height=650, grid_yside="left", title_text= "Top "+str(k)+" Corporate Customers by Orders", title_font_size=22)

                    
                    st.plotly_chart(fig)

                with cust7:

                    # title3 = "Top "+str(k)+" Profitable Customers"
                    # st.markdown(f"<h2 style='text-align: left; color: DarkBlue;'>{title3}</h2>", unsafe_allow_html=True)

                    top_customers_sub_cat =round(filtered_dates[filtered_dates['Segment'] == 'Corporate'].groupby(by=["Sub-Category"])["Quantity"].sum()) #revenue of top k customers
                    top_customers_sub_cat = pd.DataFrame(top_customers_sub_cat)
                    top_customers_sub_cat = top_customers_sub_cat.sort_values(by=['Quantity'], ascending=False)[0:i]
                    top_customers_sub_cat.reset_index(level=0, inplace=True)

                    fig = go.Figure(data=[go.Table(
                            header=dict(values=['Customer', 'Quantity'],
                            #line_color='darkslategray',
                            align='center',font=dict(color='black', size=16)),
                            cells=dict(values=[top_customers_sub_cat['Sub-Category'], # 1st column
                            top_customers_sub_cat['Quantity']], # 2nd column
                            align='center',font=dict(color='black', size=13)))])
                    
                    fig.update_layout(width=600, height=650, grid_yside="left", title_text= "Top "+str(i)+" Bought Sub-Categories by Corporate", title_font_size=22)

                    
                    st.plotly_chart(fig)

                with cust8:

                    # title3 = "Top "+str(k)+" Profitable Customers"
                    # st.markdown(f"<h2 style='text-align: left; color: DarkBlue;'>{title3}</h2>", unsafe_allow_html=True)

                    top_customers_items =round(filtered_dates[filtered_dates['Segment'] == 'Corporate'].groupby(by=["Product Name"])["Quantity"].count()) #revenue of top k customers
                    top_customers_items = pd.DataFrame(top_customers_items)
                    top_customers_items = top_customers_items.sort_values(by=['Quantity'], ascending=False)[0:i]
                    top_customers_items.reset_index(level=0, inplace=True)

                    fig = go.Figure(data=[go.Table(
                            header=dict(values=['Product Name', 'Quantity'],
                            #line_color='darkslategray',
                            align='center',font=dict(color='black', size=16)),
                            cells=dict(values=[top_customers_items['Product Name'], # 1st column
                            top_customers_items['Quantity']], # 2nd column
                            align='center',font=dict(color='black', size=13)))])
                    
                    fig.update_layout(width=600, height=650, grid_yside="left", title_text= "Top "+str(i)+" Bought Products by Corporate Customers", title_font_size=22)

                    
                    st.plotly_chart(fig)
                

            if purpose2 == 'Home Office':

                k = st.sidebar.slider("How Many Top Customer?",20, 1, 20)
                i = st.sidebar.slider("How Many Top Products?",20, 1, 20)

                # st.header('Customers')
                cust1, cust2, cust3, cust4 = st.beta_columns(4)

                with cust1:
                    title = "Total Customers"
                    st.markdown(f"<h2 style='text-align: center; color: DarkBlue;'>{title}</h2>", unsafe_allow_html=True)

                    number_of_customers = filtered_dates[filtered_dates['Segment'] == 'Home Office']['Customer ID'].nunique() #number of unique customers
                    st.markdown(f"<h2 style='text-align: center; color: red;'>{number_of_customers}</h2>", unsafe_allow_html=True)

                with cust2:

                    before_selected_date = data[data["Order Date"] < start_date]

                    title1 = "New Customers"
                    st.markdown(f"<h2 style='text-align: center; color: DarkBlue;'>{title1}</h2>", unsafe_allow_html=True)

                    new_customers = filtered_dates[(filtered_dates['Segment'] == 'Home Office') & (filtered_dates['Customer ID'].isin(before_selected_date['Customer ID'])== False)]['Customer ID'].nunique() 
                    st.markdown(f"<h2 style='text-align: center; color: red;'>{new_customers}</h2>", unsafe_allow_html=True)

                with cust3:
                    title2 = "Items Bought"
                    st.markdown(f"<h2 style='text-align: center; color: DarkBlue;'>{title2}</h2>", unsafe_allow_html=True)

                    number_of_items = filtered_dates[filtered_dates['Segment'] == 'Home Office']['Quantity'].sum() 
                    st.markdown(f"<h2 style='text-align: center; color: red;'>{number_of_items}</h2>", unsafe_allow_html=True)

                with cust4:
                    title3 = "Profit"
                    st.markdown(f"<h2 style='text-align: center; color: DarkBlue;'>{title3}</h2>", unsafe_allow_html=True)

                    profit = round(filtered_dates[filtered_dates['Segment'] == 'Home Office']['Profit'].sum()) 
                    st.markdown(f"<h2 style='text-align: center; color: red;'>{profit}</h2>", unsafe_allow_html=True)


                cust7, cust8= st.beta_columns(2)

                with cust7:
                    st.text('')
                    st.text('')
                    st.text('')

                    # title3 = "Top "+str(k)+" Profitable Customers"
                    # st.markdown(f"<h2 style='text-align: left; color: DarkBlue;'>{title3}</h2>", unsafe_allow_html=True)

                    top_customers_profits =round(filtered_dates[filtered_dates['Segment'] == 'Home Office'].groupby(by=["Customer Name"])["Profit"].sum()) #revenue of top k customers
                    top_customers_profits = pd.DataFrame(top_customers_profits)
                    top_customers_profits = top_customers_profits.sort_values(by=['Profit'], ascending=False)[0:k]
                    top_customers_profits.reset_index(level=0, inplace=True)

                    fig = go.Figure(data=[go.Table(
                            header=dict(values=['Customer', 'Profit'],
                            #line_color='darkslategray',
                            align='center',font=dict(color='black', size=16)),
                            cells=dict(values=[top_customers_profits['Customer Name'], # 1st column
                            top_customers_profits['Profit']], # 2nd column
                            align='center',font=dict(color='black', size=13)))])
                    
                    fig.update_layout(width=600, height=650, grid_yside="left", title_text= "Top "+str(k)+" Profitable Home Office Customers", title_font_size=22)

                    
                    st.plotly_chart(fig)

                with cust8:
                    st.text('')
                    st.text('')
                    st.text('')

                    # title3 = "Top "+str(k)+" Profitable Customers"
                    # st.markdown(f"<h2 style='text-align: left; color: DarkBlue;'>{title3}</h2>", unsafe_allow_html=True)

                    top_customers_orders =round(filtered_dates[filtered_dates['Segment'] == 'Home Office'].groupby(by=["Customer Name"])["Order ID"].count()) #revenue of top k customers
                    top_customers_orders = pd.DataFrame(top_customers_orders)
                    top_customers_orders = top_customers_orders.sort_values(by=['Order ID'], ascending=False)[0:k]
                    top_customers_orders.reset_index(level=0, inplace=True)

                    fig = go.Figure(data=[go.Table(
                            header=dict(values=['Customer', 'Orders'],
                            #line_color='darkslategray',
                            align='center',font=dict(color='black', size=16)),
                            cells=dict(values=[top_customers_orders['Customer Name'], # 1st column
                            top_customers_orders['Order ID']], # 2nd column
                            align='center',font=dict(color='black', size=13)))])
                    
                    fig.update_layout(width=600, height=650, grid_yside="left", title_text= "Top "+str(k)+" Home Office Customers by Orders", title_font_size=22)

                    
                    st.plotly_chart(fig)

                with cust7:

                    # title3 = "Top "+str(k)+" Profitable Customers"
                    # st.markdown(f"<h2 style='text-align: left; color: DarkBlue;'>{title3}</h2>", unsafe_allow_html=True)

                    top_customers_sub_cat =round(filtered_dates[filtered_dates['Segment'] == 'Home Office'].groupby(by=["Sub-Category"])["Quantity"].sum()) #revenue of top k customers
                    top_customers_sub_cat = pd.DataFrame(top_customers_sub_cat)
                    top_customers_sub_cat = top_customers_sub_cat.sort_values(by=['Quantity'], ascending=False)[0:i]
                    top_customers_sub_cat.reset_index(level=0, inplace=True)

                    fig = go.Figure(data=[go.Table(
                            header=dict(values=['Customer', 'Quantity'],
                            #line_color='darkslategray',
                            align='center',font=dict(color='black', size=16)),
                            cells=dict(values=[top_customers_sub_cat['Sub-Category'], # 1st column
                            top_customers_sub_cat['Quantity']], # 2nd column
                            align='center',font=dict(color='black', size=13)))])
                    
                    fig.update_layout(width=600, height=650, grid_yside="left", title_text= "Top "+str(i)+" Bought Sub-Categories by Home Office", title_font_size=22)

                    
                    st.plotly_chart(fig)

                with cust8:

                    # title3 = "Top "+str(k)+" Profitable Customers"
                    # st.markdown(f"<h2 style='text-align: left; color: DarkBlue;'>{title3}</h2>", unsafe_allow_html=True)

                    top_customers_items =round(filtered_dates[filtered_dates['Segment'] == 'Home Office'].groupby(by=["Product Name"])["Quantity"].count()) #revenue of top k customers
                    top_customers_items = pd.DataFrame(top_customers_items)
                    top_customers_items = top_customers_items.sort_values(by=['Quantity'], ascending=False)[0:i]
                    top_customers_items.reset_index(level=0, inplace=True)

                    fig = go.Figure(data=[go.Table(
                            header=dict(values=['Product Name', 'Quantity'],
                            #line_color='darkslategray',
                            align='center',font=dict(color='black', size=16)),
                            cells=dict(values=[top_customers_items['Product Name'], # 1st column
                            top_customers_items['Quantity']], # 2nd column
                            align='center',font=dict(color='black', size=13)))])
                    
                    fig.update_layout(width=600, height=650, grid_yside="left", title_text= "Top "+str(i)+" Bought Products by Home Office Customers", title_font_size=22)

                    
                    st.plotly_chart(fig)

        if purpose1 == 'Sales by Locations': 

            date = st.sidebar.date_input("Date Range", [dt.date(2011, 5, 14), dt.date(2014, 12, 31)])
            purpose1 = st.sidebar.radio('Choose an Option', ['All Categoreis', 'Technology', 'Furniture', 'Office Supplies'])

            if purpose1 == 'All Categoreis':

                data['Order Date'] = pd.to_datetime(data['Order Date'])

                start_date = pd.to_datetime(date[0])
                end_date = pd.to_datetime(date[1])

                after_start_date = data["Order Date"] >= start_date
                before_end_date = data["Order Date"] <= end_date

                between_two_dates = after_start_date & before_end_date
                filtered_dates = data.loc[between_two_dates]


                m = st.sidebar.slider("Set a Sales Threshold to Limit the Number of States to Show !", 0, 500000, 60000, step = 10000)

                filtered_dates2 = filtered_dates.groupby('State').filter(lambda g: g.Sales.sum() > m)

                temp = filtered_dates2[['State','City','Sales']].groupby(['State','City'])['Sales'].sum().reset_index()
                fig = px.treemap(temp,path=['State','City'], values='Sales')
                fig.update_layout(height=2000, width = 1000, title='Sales by States and Cities (Top 60 States)', title_font_size=25)
                                #color_discrete_sequence = px.colors.qualitative.Plotly)
                fig.data[0].textinfo = 'label+text+value'

                st.plotly_chart(fig)

            if purpose1 == 'Technology':

                data['Order Date'] = pd.to_datetime(data['Order Date'])

                start_date = pd.to_datetime(date[0])
                end_date = pd.to_datetime(date[1])

                after_start_date = data["Order Date"] >= start_date
                before_end_date = data["Order Date"] <= end_date

                between_two_dates = after_start_date & before_end_date
                filtered_dates = data.loc[between_two_dates]
                
                filtered_dates = filtered_dates[filtered_dates['Category'] == 'Technology']

                m = st.sidebar.slider("Set a Sales Threshold to Limit the Number of States to Show !", 0, 200000, 30000, step = 5000)

                filtered_dates2 = filtered_dates.groupby('State').filter(lambda g: g.Sales.sum() > m)

                temp = filtered_dates2[['State','City','Sales']].groupby(['State','City'])['Sales'].sum().reset_index()
                fig = px.treemap(temp,path=['State','City'], values='Sales')
                fig.update_layout(height=2000, width = 1000, title='Sales by States and Cities (Top 60 States)', title_font_size=25)
                                #color_discrete_sequence = px.colors.qualitative.Plotly)
                fig.data[0].textinfo = 'label+text+value'

                st.plotly_chart(fig)

            if purpose1 == 'Furniture':

                data['Order Date'] = pd.to_datetime(data['Order Date'])

                start_date = pd.to_datetime(date[0])
                end_date = pd.to_datetime(date[1])

                after_start_date = data["Order Date"] >= start_date
                before_end_date = data["Order Date"] <= end_date

                between_two_dates = after_start_date & before_end_date
                filtered_dates = data.loc[between_two_dates]
                
                filtered_dates = filtered_dates[filtered_dates['Category'] == 'Furniture']

                m = st.sidebar.slider("Set a Sales Threshold to Limit the Number of States to Show !", 0, 100000, 20000, step = 5000)

                filtered_dates2 = filtered_dates.groupby('State').filter(lambda g: g.Sales.sum() > m)

                temp = filtered_dates2[['State','City','Sales']].groupby(['State','City'])['Sales'].sum().reset_index()
                fig = px.treemap(temp,path=['State','City'], values='Sales')
                fig.update_layout(height=2000, width = 1000, title='Sales by States and Cities (Top 60 States)', title_font_size=25)
                                #color_discrete_sequence = px.colors.qualitative.Plotly)
                fig.data[0].textinfo = 'label+text+value'

                st.plotly_chart(fig)

            if purpose1 == 'Office Supplies':

                data['Order Date'] = pd.to_datetime(data['Order Date'])

                start_date = pd.to_datetime(date[0])
                end_date = pd.to_datetime(date[1])

                after_start_date = data["Order Date"] >= start_date
                before_end_date = data["Order Date"] <= end_date

                between_two_dates = after_start_date & before_end_date
                filtered_dates = data.loc[between_two_dates]
                
                filtered_dates = filtered_dates[filtered_dates['Category'] == 'Office Supplies']

                m = st.sidebar.slider("Set a Sales Threshold to Limit the Number of States to Show !", 0, 100000, 20000, step = 5000)

                filtered_dates2 = filtered_dates.groupby('State').filter(lambda g: g.Sales.sum() > m)

                temp = filtered_dates2[['State','City','Sales']].groupby(['State','City'])['Sales'].sum().reset_index()
                fig = px.treemap(temp,path=['State','City'], values='Sales')
                fig.update_layout(height=2000, width = 1000, title='Sales by States and Cities (Top 60 States)', title_font_size=25)
                                #color_discrete_sequence = px.colors.qualitative.Plotly)
                fig.data[0].textinfo = 'label+text+value'

                st.plotly_chart(fig)


elif purpose == "Sales Forecasting":
    
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title('Sales Forecasting')
    st.text('')
    st.text('')

    purpose1 = st.sidebar.radio("Which Category's Sales do you want to Forecast?", ['All Categories', 'Technology', 'Furniture', 'Office Supplies'])
    
    if purpose1 == 'All Categories':

        m = st.sidebar.slider("For How Many Months in Advance do you want to Forecast?", 2, 120)

        data['Order Date'] = pd.to_datetime(data['Order Date'])
        data = data.groupby('Order Date')['Sales'].sum().reset_index()
        data.sort_values('Order Date')
        data = data.set_index('Order Date')
        y = data['Sales'].resample('MS').mean()
        
        # Training the ARIMA model.
        model = sm.tsa.statespace.SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_invertibility=False)

        results = model.fit()
        #print(results.summary().tables[1])

        pred_uc = results.get_forecast(steps=m)
        pred_ci = pred_uc.conf_int()

        st.header('Current and Forecasted Sales')

        ax = y.plot(label='observed', figsize=(14, 7))
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        #ax.set_title('Current and Forecasted Sales')

        plt.legend()
        plt.show()

        st.pyplot()

        forecasts = pd.DataFrame(pred_uc.predicted_mean).reset_index()
        forecasts.rename(columns = {'index' : 'Month', 'predicted_mean' : 'Forecasted Sales'}, inplace = True)

        st.text('')
        st.text('')

        st.header('Forecasted Monthly Sales')

        st.table(forecasts[0:m])

    if purpose1 == 'Technology':

        data = data[data['Category'] == 'Technology']

        m = st.sidebar.slider("For How Many Months in Advance do you want to Forecast?", 2, 120)

        data['Order Date'] = pd.to_datetime(data['Order Date'])
        data = data.groupby('Order Date')['Sales'].sum().reset_index()
        data.sort_values('Order Date')
        data = data.set_index('Order Date')
        y = data['Sales'].resample('MS').mean()
        
        # Training the ARIMA model.
        model = sm.tsa.statespace.SARIMAX(y, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), enforce_invertibility=False)

        results = model.fit()
        #print(results.summary().tables[1])

        pred_uc = results.get_forecast(steps=m)
        pred_ci = pred_uc.conf_int()

        st.header('Current and Forecasted Sales for the Technology Category')

        ax = y.plot(label='observed', figsize=(14, 7))
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        #ax.set_title('Current and Forecasted Sales')

        plt.legend()
        plt.show()

        st.pyplot()

        forecasts = pd.DataFrame(pred_uc.predicted_mean).reset_index()
        forecasts.rename(columns = {'index' : 'Month', 'predicted_mean' : 'Forecasted Sales'}, inplace = True)

        st.text('')
        st.text('')

        st.header('Forecasted Monthly Sales for the Technology Category')

        st.table(forecasts[0:m])


    if purpose1 == 'Furniture':

        data = data[data['Category'] == 'Furniture']

        m = st.sidebar.slider("For How Many Months in Advance do you want to Forecast?", 2, 120)

        data['Order Date'] = pd.to_datetime(data['Order Date'])
        data = data.groupby('Order Date')['Sales'].sum().reset_index()
        data.sort_values('Order Date')
        data = data.set_index('Order Date')
        y = data['Sales'].resample('MS').mean()
        
        # Training the ARIMA model.
        model = sm.tsa.statespace.SARIMAX(y, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), enforce_invertibility=False)

        results = model.fit()
        #print(results.summary().tables[1])

        pred_uc = results.get_forecast(steps=m)
        pred_ci = pred_uc.conf_int()

        st.header('Current and Forecasted Sales for the Furniture Category')

        ax = y.plot(label='observed', figsize=(14, 7))
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        #ax.set_title('Current and Forecasted Sales')

        plt.legend()
        plt.show()

        st.pyplot()

        forecasts = pd.DataFrame(pred_uc.predicted_mean).reset_index()
        forecasts.rename(columns = {'index' : 'Month', 'predicted_mean' : 'Forecasted Sales'}, inplace = True)

        st.text('')
        st.text('')

        st.header('Forecasted Monthly Sales for the Furniture Category')

        st.table(forecasts[0:m])

    if purpose1 == 'Office Supplies':

        data = data[data['Category'] == 'Office Supplies']

        m = st.sidebar.slider("For How Many Months in Advance do you want to Forecast?", 2, 120)

        data['Order Date'] = pd.to_datetime(data['Order Date'])
        data = data.groupby('Order Date')['Sales'].sum().reset_index()
        data.sort_values('Order Date')
        data = data.set_index('Order Date')
        y = data['Sales'].resample('MS').mean()
        
        # Training the ARIMA model.
        model = sm.tsa.statespace.SARIMAX(y, order=(0, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_invertibility=False)

        results = model.fit()
        #print(results.summary().tables[1])

        pred_uc = results.get_forecast(steps=m)
        pred_ci = pred_uc.conf_int()

        st.header('Current and Forecasted Sales for the Office Supplies Category')

        ax = y.plot(label='observed', figsize=(14, 7))
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        #ax.set_title('Current and Forecasted Sales')

        plt.legend()
        plt.show()

        st.pyplot()

        forecasts = pd.DataFrame(pred_uc.predicted_mean).reset_index()
        forecasts.rename(columns = {'index' : 'Month', 'predicted_mean' : 'Forecasted Sales'}, inplace = True)

        st.text('')
        st.text('')

        st.header('Forecasted Monthly Sales for the Office Supplies Category')

        st.table(forecasts[0:m])


elif purpose == 'RFM Analysis and Segmentation':
    st.title('RFM Analysis and Customer Segmentation')
    st.text('')

    segments = data['Segment'].unique()
    purpose1 = st.sidebar.radio("Which Customer Group do you want to Analyze?", segments)
    

    data = data.dropna()
    data['Customer ID'] = data['Customer ID'].astype(str)
    data['Order Date'] = pd.to_datetime(data['Order Date']).dt.date
    data['Date'] = data['Order Date']
    raw_df = data.sort_values(by=['Date'])
    raw_df = raw_df.set_index('Date')
    
    #createing two date points for filtering dataframe
    end_date = raw_df.index.max()
    
    #let's say we would like to pick the period of the last 90 days
    start_date = end_date - timedelta(days=120) 

    snapshot_date = end_date + timedelta(days=1) 
    
    st.write("RFM model in particular is at the core of customer segmentation. RFM studies customers’ behaviour and cluster them by using three metrics:")
    st.write("   1. Recency (R): measure the number of days since the last purchase to a hypothetical snapshot day.")
    st.write("   2. Frequency (F): measure the number of transaction made during the period of study.")
    st.write("   3. Monetary Value (M): measure how much money each customer has spent during the period of study.")

    def get_table_download_link_csv(df):
        #csv = df.to_csv(index=False)
        csv = df.to_csv().encode()
        #b64 = base64.b64encode(csv.encode()).decode() 
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="captura.csv" target="_blank">Download csv file</a>'
        return href

    if purpose1 == 'Consumer':
        raw_df = raw_df[raw_df['Segment'] == 'Consumer']
        study_df = raw_df[start_date : end_date]
        
        data_rfm = study_df.groupby(['Customer Name']).agg({
            'Order Date': lambda x : (snapshot_date - x.max()).days,
            'Order ID':'nunique',
            'Sales':'sum'
        })

        data_rfm = data_rfm.rename(columns={
            'Order Date':'Recency',
            'Order ID':'Frequency',
            'Sales': 'MonetaryValue'
        })
       
        data_rfm_log = np.log(data_rfm) 

        scaler = StandardScaler()
        scaler.fit(data_rfm_log)
        #store it separately for clustering
        data_rfm_standard = scaler.transform(data_rfm_log)
        #turn the processed data back into a dataframe
        data_rfm_standard = pd.DataFrame(data = data_rfm_standard, 
                                        index = data_rfm.index, 
                                        columns = data_rfm.columns) 

        kmeans = KMeans(n_clusters = 3, random_state=1)
        #compute k-means clustering on pre-processed data
        kmeans.fit(data_rfm_standard)
        #extract cluster labels from labels_ attribute
        cluster_labels = kmeans.labels_ 
        
        df_k3 = data_rfm.assign(Cluster = cluster_labels) 
        summary = df_k3.groupby(['Cluster']).agg({
        'Recency':'mean',
        'Frequency':'mean',
        'MonetaryValue':['mean']})

        df_k3['Cluster'] = df_k3['Cluster'].replace({0: 'Least Important', 2: 'Most Important', 1: 'Important'})
        summary = summary.rename(index={0: 'Least Important', 2: 'Most Important', 1: 'Important'})
        
        st.text('')
        st.text('')

        st.header('RFM Analysis')
        st.table(summary)

        st.text('')

        m = st.sidebar.slider("For How Many Top Customer do you want to See?", 1, 50)

        df_k3_least = df_k3[df_k3['Cluster'] == 'Least Important']
        df_k3_least1 = df_k3_least[['Recency', 'Frequency', 'MonetaryValue']].sort_values(by='MonetaryValue', ascending = False)[0:m]

        df_k3_imp = df_k3[df_k3['Cluster'] == 'Important']
        df_k3_imp1 = df_k3_imp[['Recency', 'Frequency', 'MonetaryValue']].sort_values(by='MonetaryValue', ascending = False)[0:m]
        
        
        df_k3_most = df_k3[df_k3['Cluster'] == 'Most Important']
        df_k3_most1 = df_k3_most[['Recency', 'Frequency', 'MonetaryValue']].sort_values(by='MonetaryValue', ascending = False)[0:m]
        
        st.header('The Most Important Customer Segment')
        st.table(df_k3_most1) 
        st.markdown(get_table_download_link_csv(df_k3_most), unsafe_allow_html=True)
        st.text('')


        st.header('The Second Most Important Customer Segment')
        st.table(df_k3_imp1)
        st.markdown(get_table_download_link_csv(df_k3_imp), unsafe_allow_html=True)
        st.text('')

        st.header('The Third Most Important Customer Segment')
        st.table(df_k3_least1)
        st.markdown(get_table_download_link_csv(df_k3_least), unsafe_allow_html=True)
        st.text('')

        st.header('Visualizing the Customer Clusters')

        fig = px.scatter_3d(df_k3, x='Recency', 
                   y='Frequency', z='MonetaryValue', 
                   color='Cluster')

        fig.update_layout(width=1000, height=800)

        st.plotly_chart(fig)


    if purpose1 == 'Corporate':
        raw_df = raw_df[raw_df['Segment'] == 'Corporate']
        study_df = raw_df[start_date : end_date]
        
        data_rfm = study_df.groupby(['Customer Name']).agg({
            'Order Date': lambda x : (snapshot_date - x.max()).days,
            'Order ID':'nunique',
            'Sales':'sum'
        })

        data_rfm = data_rfm.rename(columns={
            'Order Date':'Recency',
            'Order ID':'Frequency',
            'Sales': 'MonetaryValue'
        })
       
        data_rfm_log = np.log(data_rfm) 

        scaler = StandardScaler()
        scaler.fit(data_rfm_log)
        #store it separately for clustering
        data_rfm_standard = scaler.transform(data_rfm_log)
        #turn the processed data back into a dataframe
        data_rfm_standard = pd.DataFrame(data = data_rfm_standard, 
                                        index = data_rfm.index, 
                                        columns = data_rfm.columns) 

        kmeans = KMeans(n_clusters = 3, random_state=1)
        #compute k-means clustering on pre-processed data
        kmeans.fit(data_rfm_standard)
        #extract cluster labels from labels_ attribute
        cluster_labels = kmeans.labels_ 
        
        df_k3 = data_rfm.assign(Cluster = cluster_labels) 
        summary = df_k3.groupby(['Cluster']).agg({
        'Recency':'mean',
        'Frequency':'mean',
        'MonetaryValue':['mean']})

        df_k3['Cluster'] = df_k3['Cluster'].replace({0: 'Least Important', 2: 'Most Important', 1: 'Important'})
        summary = summary.rename(index={0: 'Least Important', 2: 'Most Important', 1: 'Important'})
        
        st.text('')
        st.text('')

        st.header('RFM Analysis')
        st.table(summary)

        st.text('')

        m = st.sidebar.slider("For How Many Top Customer do you want to See?", 1, 50)

        df_k3_least = df_k3[df_k3['Cluster'] == 'Least Important']
        df_k3_least1 = df_k3_least[['Recency', 'Frequency', 'MonetaryValue']].sort_values(by='MonetaryValue', ascending = False)[0:m]

        df_k3_imp = df_k3[df_k3['Cluster'] == 'Important']
        df_k3_imp1 = df_k3_imp[['Recency', 'Frequency', 'MonetaryValue']].sort_values(by='MonetaryValue', ascending = False)[0:m]
        
        
        df_k3_most = df_k3[df_k3['Cluster'] == 'Most Important']
        df_k3_most1 = df_k3_most[['Recency', 'Frequency', 'MonetaryValue']].sort_values(by='MonetaryValue', ascending = False)[0:m]
        
        st.header('The Most Important Customer Segment')
        st.table(df_k3_most1) 
        st.markdown(get_table_download_link_csv(df_k3_most), unsafe_allow_html=True)
        st.text('')


        st.header('The Second Most Important Customer Segment')
        st.table(df_k3_imp1)
        st.markdown(get_table_download_link_csv(df_k3_imp), unsafe_allow_html=True)
        st.text('')

        st.header('The Third Most Important Customer Segment')
        st.table(df_k3_least1)
        st.markdown(get_table_download_link_csv(df_k3_least), unsafe_allow_html=True)
        st.text('')

        st.header('Visualizing the Customer Clusters')

        fig = px.scatter_3d(df_k3, x='Recency', 
                   y='Frequency', z='MonetaryValue', 
                   color='Cluster')

        fig.update_layout(width=1000, height=800)

        st.plotly_chart(fig)


    if purpose1 == 'Home Office':
        raw_df = raw_df[raw_df['Segment'] == 'Home Office']
        study_df = raw_df[start_date : end_date]
        
        data_rfm = study_df.groupby(['Customer Name']).agg({
            'Order Date': lambda x : (snapshot_date - x.max()).days,
            'Order ID':'nunique',
            'Sales':'sum'
        })

        data_rfm = data_rfm.rename(columns={
            'Order Date':'Recency',
            'Order ID':'Frequency',
            'Sales': 'MonetaryValue'
        })
       
        data_rfm_log = np.log(data_rfm) 

        scaler = StandardScaler()
        scaler.fit(data_rfm_log)
        #store it separately for clustering
        data_rfm_standard = scaler.transform(data_rfm_log)
        #turn the processed data back into a dataframe
        data_rfm_standard = pd.DataFrame(data = data_rfm_standard, 
                                        index = data_rfm.index, 
                                        columns = data_rfm.columns) 

        kmeans = KMeans(n_clusters = 3, random_state=1)
        #compute k-means clustering on pre-processed data
        kmeans.fit(data_rfm_standard)
        #extract cluster labels from labels_ attribute
        cluster_labels = kmeans.labels_ 
        
        df_k3 = data_rfm.assign(Cluster = cluster_labels) 
        summary = df_k3.groupby(['Cluster']).agg({
        'Recency':'mean',
        'Frequency':'mean',
        'MonetaryValue':['mean']})

        df_k3['Cluster'] = df_k3['Cluster'].replace({0: 'Least Important', 2: 'Most Important', 1: 'Important'})
        summary = summary.rename(index={0: 'Least Important', 2: 'Most Important', 1: 'Important'})
        
        st.text('')
        st.text('')

        st.header('RFM Analysis')
        st.table(summary)

        st.text('')

        m = st.sidebar.slider("For How Many Top Customer do you want to See?", 1, 50)

        df_k3_least = df_k3[df_k3['Cluster'] == 'Least Important']
        df_k3_least1 = df_k3_least[['Recency', 'Frequency', 'MonetaryValue']].sort_values(by='MonetaryValue', ascending = False)[0:m]

        df_k3_imp = df_k3[df_k3['Cluster'] == 'Important']
        df_k3_imp1 = df_k3_imp[['Recency', 'Frequency', 'MonetaryValue']].sort_values(by='MonetaryValue', ascending = False)[0:m]
        
        
        df_k3_most = df_k3[df_k3['Cluster'] == 'Most Important']
        df_k3_most1 = df_k3_most[['Recency', 'Frequency', 'MonetaryValue']].sort_values(by='MonetaryValue', ascending = False)[0:m]
        
        st.header('The Most Important Customer Segment')
        st.table(df_k3_most1) 
        st.markdown(get_table_download_link_csv(df_k3_most), unsafe_allow_html=True)
        st.text('')


        st.header('The Second Most Important Customer Segment')
        st.table(df_k3_imp1)
        st.markdown(get_table_download_link_csv(df_k3_imp), unsafe_allow_html=True)
        st.text('')

        st.header('The Third Most Important Customer Segment')
        st.table(df_k3_least1)
        st.markdown(get_table_download_link_csv(df_k3_least), unsafe_allow_html=True)
        st.text('')

        st.header('Visualizing the Customer Clusters')

        fig = px.scatter_3d(df_k3, x='Recency', 
                   y='Frequency', z='MonetaryValue', 
                   color='Cluster')

        fig.update_layout(width=1000, height=800)

        st.plotly_chart(fig)