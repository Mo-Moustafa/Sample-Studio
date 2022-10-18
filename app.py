from cmath import phase
from tkinter import CENTER, font
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import plotly


# Sidebar

menu = st.sidebar.radio('menu', options=['Main', 'Sine Wave', 'CSV'])


if menu == 'Main':
    st.title('Sampling Studio')
    st.text('This is a web app to show sampled signals')
    upload_file = st.file_uploader('Upload your signal')



# Generated Wave page

elif menu == 'Sine Wave':

    st.title("Generated Sine Wave")

    # Original Attributes

    freq = 4
    t = np.linspace(0,1,2000)
    y1 = np.sin(2 * np.pi * freq * t) 

    # Sample Attributes

    sample_freq = st.slider('Sampling Frequincy', min_value=1, max_value=100, step=1)

    s_rate = sample_freq #Sampling Frequency

    T = 1 / s_rate #How much time for a full cycle

    numberOfSamples = np.arange(0, 1/T) #Number of Samples on the graph

    nT = numberOfSamples * T #To spread the samples right on the graph

    y2 = np.sin(2 * np.pi * freq * nT)

    # Plotting Original Signal

    fig1 = plt.figure(figsize=(10,6))
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Amplitude', fontsize=15)
    plt.title("Original Signal")
    plt.plot(t, y1)
    plt.grid(True)
    plt.plot(nT, y2, 'ro') # Sample Points
    st.plotly_chart(fig1)

    # Plotting Reconstructed Signal

    fig2 = plt.figure(figsize=(10,6))
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Amplitude', fontsize=15)
    plt.title("Reconstructed Signal")
    plt.plot(nT , y2)
    plt.grid(True)
    st.plotly_chart(fig2)


# CSV Page

elif menu == 'CSV':

    st.title("CSV Signal")



    # Original Attributes

    # data = pd.read_csv('E:\Engineering\Python\EMG_Healthy_Modified_Less_Data.csv')
    # t = data['Time']
    # y1 = data['Value']

    
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type={"csv", "txt ,xlsx"})

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        df = input_df.dropna(axis=0,how='any')
        file_name= uploaded_file.name


       
        #st.write(df)  #showing the data frame
        

        time_data=df[ df.head(0).columns[0]]
        
        amplitude_data=df[ df.head(0).columns[1]]



##To show the maximum points and the lrngth of the file

        # st.write('Maximum time:',time_data.max())
        # st.write('Number of records',len(time_data))

        # st.write('Maximum value:',amplitude_data.max())
        # st.write('Number of records',len(amplitude_data))
        

##To set a title
        if 'EMG' or 'emg' in file_name:
            title = ' EMG Signal'
        elif 'ECG' or 'ecg' in file_name:
            title = ' ECG Signal'
        elif 'EEG' or 'eeg' in file_name:
            title = ' EEG Signal'
        else:
            title = ' Signal'


        # Sample Attributes

        sample_freq = st.slider('Sampling Frequincy', min_value=1, max_value=100, step=1)

        s_rate = sample_freq # sampling frequency
        T = 1 / s_rate
        numberOfSamples = 0.305/T
        calc = round(1200 / numberOfSamples)

        nT = time_data[0:1200:calc] # Spreading Samples
        y2 = amplitude_data[0:1200:calc]

        # Plotting Original Signal
        
        fig1 = plt.figure(figsize=(10,6))
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Amplitude', fontsize=15)
        plt.title(f"Original {title}")
        plt.plot(time_data,amplitude_data)
        plt.plot(nT, y2, 'ro')
        plt.grid(True)
        st.plotly_chart(fig1)

        # Plotting Reconstructed Signal

        fig2 = plt.figure(figsize=(10,6))
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Amplitude', fontsize=15)
        plt.title(f"Reconstructed {title}")
        plt.plot(nT, y2)
        plt.grid(True)
        st.plotly_chart(fig2)

    else:
        st.write('Awaiting CSV file to be uploaded.')