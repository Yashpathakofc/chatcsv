import streamlit as st
from transformers import pipeline
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

generator = pipeline('text-generation', model='gpt2')

def chat_with_csv(df, prompt):
    csv_data = df.head(30).to_csv(index=False)
    query = f"Based on the following CSV data, answer the query:\n\n{csv_data}\n\nQuery: {prompt}"
    with torch.no_grad():
        result = generator(query, max_length=150)
    return result[0]['generated_text']

def perform_statistical_analysis(df):
    statistics = {}
    numeric_df = df.select_dtypes(include=[np.number])
    
    if not numeric_df.empty:
        statistics['mean'] = numeric_df.mean().to_dict()
        statistics['median'] = numeric_df.median().to_dict()
        statistics['mode'] = numeric_df.mode().iloc[0].to_dict()
        statistics['std_dev'] = numeric_df.std().to_dict()
        correlation = numeric_df.corr()
        statistics['correlation'] = correlation
    else:
        statistics['mean'] = {}
        statistics['median'] = {}
        statistics['mode'] = {}
        statistics['std_dev'] = {}
        statistics['correlation'] = pd.DataFrame()
    
    return statistics

def plot_histogram(data, column):
    data = data.head(100)  
    plt.figure(figsize=(10, 5))
    plt.hist(data[column], bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    return plt

def plot_line_chart(data, column):
    data = data.head(100)
    plt.figure(figsize=(10, 5))
    plt.plot(data[column], label=column, marker='o', linestyle='-')
    plt.title(f'Line Chart of {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)
    return plt

def plot_line_chart_two_columns(data, x_column, y_column):
    data = data.head(100)
    plt.figure(figsize=(10, 5))
    plt.plot(data[x_column], label=x_column, marker='o', linestyle='-')
    plt.plot(data[y_column], label=y_column, marker='x', linestyle='--')
    plt.title(f'Line Chart: {x_column} and {y_column}')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    return plt

def plot_bar_graph(data, column):
    data = data.head(100)
    plt.figure(figsize=(10, 5))
    plt.bar( data[column],data.index, color='blue', edgecolor='black')
    plt.title(f'Bar Graph of {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.grid(True)
    return plt

def plot_scatter_plot(data, x_column, y_column):
    data = data.head(100)
    plt.figure(figsize=(10, 5))
    plt.scatter(data[x_column], data[y_column], color='blue', alpha=0.7)
    plt.title(f'Scatter Plot: {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)
    return plt

def generate_description(data, column):
    desc = data[column].describe()
    description = (
        f"**{column} Statistics:**\n\n"
        f"- **Count:** {desc['count']}\n"
        f"- **Mean:** {desc['mean']:.2f}\n"
        f"- **Standard Deviation:** {desc['std']:.2f}\n"
        f"- **Min:** {desc['min']:.2f}\n"
        f"- **25th Percentile:** {desc['25%']:.2f}\n"
        f"- **Median:** {desc['50%']:.2f}\n"
        f"- **75th Percentile:** {desc['75%']:.2f}\n"
        f"- **Max:** {desc['max']:.2f}\n"
    )
    
    paragraph = (
        f"The data for {column} shows a count of {desc['count']} entries. The mean value is {desc['mean']:.2f}, "
        f"with a standard deviation of {desc['std']:.2f}. The minimum recorded value is {desc['min']:.2f}, "
        f"while the maximum is {desc['max']:.2f}. The 25th percentile is {desc['25%']:.2f}, the median (or 50th percentile) "
        f"is {desc['50%']:.2f}, and the 75th percentile is {desc['75%']:.2f}. This gives a comprehensive overview of the distribution "
        f"and spread of {column} in the dataset."
    )
    
    return description + "\n\n" + paragraph

def generate_description_two_columns(data, x_column, y_column):
    desc_x = data[x_column].describe()
    desc_y = data[y_column].describe()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### **{x_column} Statistics:**")
        st.markdown(f"- **Count:** {desc_x['count']:.0f}")
        st.markdown(f"- **Mean:** {desc_x['mean']:.2f}")
        st.markdown(f"- **Standard Deviation:** {desc_x['std']:.2f}")
        st.markdown(f"- **Min:** {desc_x['min']:.2f}")
        st.markdown(f"- **25th Percentile:** {desc_x['25%']:.2f}")
        st.markdown(f"- **Median:** {desc_x['50%']:.2f}")
        st.markdown(f"- **75th Percentile:** {desc_x['75%']:.2f}")
        st.markdown(f"- **Max:** {desc_x['max']:.2f}")

    with col2:
        st.markdown(f"### **{y_column} Statistics:**")
        st.markdown(f"- **Count:** {desc_y['count']:.0f}")
        st.markdown(f"- **Mean:** {desc_y['mean']:.2f}")
        st.markdown(f"- **Standard Deviation:** {desc_y['std']:.2f}")
        st.markdown(f"- **Min:** {desc_y['min']:.2f}")
        st.markdown(f"- **25th Percentile:** {desc_y['25%']:.2f}")
        st.markdown(f"- **Median:** {desc_y['50%']:.2f}")
        st.markdown(f"- **75th Percentile:** {desc_y['75%']:.2f}")
        st.markdown(f"- **Max:** {desc_y['max']:.2f}")


    paragraph = (
        
        f"The data for **{x_column}** shows a count of {desc_x['count']} entries. The mean value is {desc_x['mean']:.2f}, "
        f"with a standard deviation of {desc_x['std']:.2f}. The minimum recorded value is {desc_x['min']:.2f}, "
        f"while the maximum is {desc_x['max']:.2f}. The 25th percentile is {desc_x['25%']:.2f}, the median (or 50th percentile) "
        f"is {desc_x['50%']:.2f}, and the 75th percentile is {desc_x['75%']:.2f}. This provides a comprehensive overview of the distribution "
        f"and spread of **{x_column}** in the dataset.\n\n"
        f"Similarly, for **{y_column}**, the data shows a count of {desc_y['count']} entries. The mean value is {desc_y['mean']:.2f}, "
        f"with a standard deviation of {desc_y['std']:.2f}. The minimum recorded value is {desc_y['min']:.2f}, "
        f"while the maximum is {desc_y['max']:.2f}. The 25th percentile is {desc_y['25%']:.2f}, the median (or 50th percentile) "
        f"is {desc_y['50%']:.2f}, and the 75th percentile is {desc_y['75%']:.2f}. This provides a comprehensive overview of the distribution "
        f"and spread of **{y_column}** in the dataset."
    )

    st.markdown(paragraph)

st.set_page_config(layout='wide')
st.markdown("""
    <h1 style='text-align: center; font-size: 2.5em;'>📊 ChatCSV <small style='font-size: 0.7em;'>powered by LLM</small></h1>
""", unsafe_allow_html=True)

input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

if input_csv is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.info("CSV Uploaded Successfully")
        data = pd.read_csv(input_csv)
        st.dataframe(data, use_container_width=True)
        
        if st.checkbox("Show basic statistics"):
            stats = perform_statistical_analysis(data)
            st.subheader("Statistical Analysis")
            
            with st.expander("Mean"):
                if stats['mean']:
                    st.write(stats['mean'])
                else:
                    st.write("No numeric columns found.")
                
            with st.expander("Median"):
                if stats['median']:
                    st.write(stats['median'])
                else:
                    st.write("No numeric columns found.")
                
            with st.expander("Mode"):
                if stats['mode']:
                    st.write(stats['mode'])
                else:
                    st.write("No numeric columns found.")
                
            with st.expander("Standard Deviation"):
                if stats['std_dev']:
                    st.write(stats['std_dev'])
                else:
                    st.write("No numeric columns found.")
                
            with st.expander("Correlation Coefficient"):
                if not stats['correlation'].empty:
                    st.dataframe(stats['correlation'])
                else:
                    st.write("No numeric columns found.")
    
    with col2:
        st.info("Ask your queries")
        
        input_text = st.text_area("Enter your query")
        if st.button("Submit"):
            if input_text:
                st.info("Your Query: " + input_text)
                query = input_text.lower()
                
                if 'histogram' in query:
                    column_name = query.split('histogram of ')[-1].strip()
                    matched_columns = [col for col in data.columns if col.lower() == column_name]
                    if matched_columns:
                        column_name = matched_columns[0]
                        fig = plot_histogram(data, column_name)
                        st.pyplot(fig)
                        st.markdown(generate_description(data, column_name))
                    else:
                        st.error(f"Column '{column_name}' not found in the CSV.")
                
                elif 'line chart' in query:
                    if ' and ' in query:
                        columns = query.split('line chart of ')[-1].strip().split(' and ')
                        if len(columns) == 2:
                            x_column, y_column = columns[0].strip(), columns[1].strip()
                            matched_x_columns = [col for col in data.columns if col.lower() == x_column]
                            matched_y_columns = [col for col in data.columns if col.lower() == y_column]
                            if matched_x_columns and matched_y_columns:
                                x_column, y_column = matched_x_columns[0], matched_y_columns[0]
                                fig = plot_line_chart_two_columns(data, x_column, y_column)
                                st.pyplot(fig)
                                st.markdown(generate_description_two_columns(data, x_column, y_column))
                            else:
                                st.error(f"Columns '{x_column}' or '{y_column}' not found in the CSV.")
                        else:
                            st.error("Invalid query format for line chart. Please use 'Line Chart of X and Y'.")
                    else:
                        column_name = query.split('line chart of ')[-1].strip()
                        matched_columns = [col for col in data.columns if col.lower() == column_name]
                        if matched_columns:
                            column_name = matched_columns[0]
                            fig = plot_line_chart(data, column_name)
                            st.pyplot(fig)
                            st.markdown(generate_description(data, column_name))
                        else:
                            st.error(f"Column '{column_name}' not found in the CSV.")
                
                elif 'bar graph' in query:
                    column_name = query.split('bar graph of ')[-1].strip()
                    matched_columns = [col for col in data.columns if col.lower() == column_name]
                    if matched_columns:
                        column_name = matched_columns[0]
                        fig = plot_bar_graph(data, column_name)
                        st.pyplot(fig)
                        st.markdown(generate_description(data, column_name))
                    else:
                        st.error(f"Column '{column_name}' not found in the CSV.")
                
                elif 'scatter plot' in query:
                    columns = query.split('scatter plot of ')[-1].strip().split(' vs ')
                    if len(columns) == 2:
                        x_column, y_column = columns[0].strip(), columns[1].strip()
                        matched_x_columns = [col for col in data.columns if col.lower() == x_column]
                        matched_y_columns = [col for col in data.columns if col.lower() == y_column]
                        if matched_x_columns and matched_y_columns:
                            x_column, y_column = matched_x_columns[0], matched_y_columns[0]
                            fig = plot_scatter_plot(data, x_column, y_column)
                            st.pyplot(fig)
                            st.markdown(generate_description_two_columns(data, x_column, y_column))
                        else:
                            st.error(f"Columns '{x_column}' or '{y_column}' not found in the CSV.")
                    else:
                        st.error("Invalid query format for scatter plot. Please use 'Scatter Plot of X vs Y'.")
                
                else:
                    result = chat_with_csv(data, input_text)
                    st.success(result)
            else:
                st.warning("Please enter a query.")
st.markdown("<footer style='text-align: center; margin-top: 50px;'><h2>Yash Pathak</h2><div>All rights reserved 2024<div></footer>", unsafe_allow_html=True)