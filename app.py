from cmath import phase
from math import ceil
from re import X
from tkinter import CENTER, font
from turtle import color
from turtle import title
from lib2to3.pgen2.token import EQEQUAL
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import plotly
import csv
import os



def plot(signal='',x1=[],y1=[],x2=[],y2=[], x3=[] , y3=[], interp = False):

    fig = plt.figure(figsize=(1, 5))
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Amplitude', fontsize=15)
    plt.title(signal)
    plt.plot(x1, y1)
    plt.grid(True)

    if len(x2) != 0 and len(y2) != 0:
        plt.plot(x2, y2, 'ro')  # Sampling points

    if (s_Interpolation):
        plt.plot(x3, y3, '--') # Sampling interpulation

    st.plotly_chart(fig,use_container_width=True)





# Sidebar

menu = st.sidebar.radio('menu', options=['Main', 'Sine Wave', 'CSV'])

if menu == 'Main':
    st.title('Sampling Studio')
    st.text('This is a web app to show sampled signals')

# Generated Wave page

elif menu == 'Sine Wave':

    st.title("Generated Sine Wave")
    
    # Original Attributes

    freq = 4
    t = np.arange(0,1,0.001)
    y1 = np.sin(2 * np.pi * freq * t) 

    # Sample Attributes

    sample_freq = st.slider('Sampling Frequincy', min_value=1, max_value=100, step=1)

    s_Interpolation = st.checkbox('Show Interpolation')

    s_rate = sample_freq    #Sampling Frequency

    Ts = 1 / s_rate  #How much time for a full cycle

    nT = np.arange(0,1,Ts)    #To spread the samples right on the graph

    y2 = np.sin(2 * np.pi * freq * nT) # Resampled


    # Sinc Interpolation

    y_reconstruction = np.zeros(len(t))
    for i in range (1,len(t)):
        for n in range (1,len(nT)):
            y_reconstruction[i] += y2[n] * np.sinc((t[i]-nT[n])/Ts)

    # Plotting

    plot(f"Original Signal", t, y1, nT, y2, t, y_reconstruction, s_Interpolation)     # Plotting Original Signal
    plot(f"Reconstructed Signal",t , y_reconstruction)     # Plotting Reconstructed Signal


# CSV Page

elif menu == 'CSV':

    st.title("Upload Signal")

    # Upload CSV

    uploaded_file = st.file_uploader("Upload your input CSV file", type={"csv", "txt ,xlsx"})

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        df = input_df.dropna(axis=0,how='any')
        file_name= uploaded_file.name

        # Read data

        time_data=df[ df.head(0).columns[0]]
        amplitude_data=df[ df.head(0).columns[1]]
        time_maximum = time_data.max()
        time_minimum = time_data.min()
        numberOfRecords = len(time_data)
        
        # Sample Attributes

        sample_freq = st.slider('Sampling Frequincy', min_value=1, max_value=100, step=1)

        s_Interpolation = st.checkbox('Show Interpolation')

        s_rate = sample_freq # sampling frequency
        T = 1 / s_rate

        numberOfSamples = (time_maximum - time_minimum)/T

        calc = ceil(numberOfRecords / numberOfSamples)

        nT = time_data[0:numberOfRecords:calc] # Spreading Samples
        y2 = amplitude_data[0:numberOfRecords:calc]

        # Sinc Interpolation
        y_reconstruction = np.zeros(len(time_data))
        for i in range (0,len(time_data)-1):
            for x,y in zip(nT,y2):
                y_reconstruction[i] += y * np.sinc((time_data[i]-x)/T)

        # Plotting
        
        plot(f"Original Signal",time_data,amplitude_data,nT,y2, time_data, y_reconstruction, s_Interpolation)  # Plotting Original Signal
        plot(f"Reconstructed Signal",time_data,y_reconstruction)  # Plotting Reconstructed Signal

    else:
        st.write('Awaiting CSV file to be uploaded.')