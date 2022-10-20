from cmath import phase
from tkinter import CENTER, font
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import plotly
from bokeh.plotting import figure as fg






def signal_ploting_function(titel='',x1=[],y1=[],x2=[],y2=[]):

            line1_timedata=x1
            line1_valdata=y1
            line2_timedata=x2
            line2_valdata=y2

            
            figure = fg(
            title=titel,
            x_axis_label='Time',
            y_axis_label='Amplitude',
            tools="pan,wheel_zoom,box_zoom,reset   ,undo ,redo ,save "
            )

            figure.line(line1_timedata, line1_valdata,legend_label='signal',line_color='blue',line_width=2)

            if len(line2_timedata) != 0 and len(line2_valdata) != 0:
             figure.circle(line2_timedata,line2_valdata,legend_label='sample points',color='red', size=5)

            figure.sizing_mode = 'scale_both'
            figure.xgrid.grid_line_color = 'darkgray'
            figure.ygrid.grid_line_color = 'darkgray'
            st.bokeh_chart(figure, use_container_width=True)
            
        

st.set_page_config(layout='wide')


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

    # fig1 = plt.figure(figsize=(10,6))
    # plt.xlabel('Time', fontsize=15)
    # plt.ylabel('Amplitude', fontsize=15)
    # plt.title("Original Signal")
    # plt.plot(t, y1)
    # plt.grid(True)
    # plt.plot(nT, y2, 'ro') # Sample Points
    # st.plotly_chart(fig1)
    signal_ploting_function(f"Original Signal",t,y1,nT,y2)

    # Plotting Reconstructed Signal

    # fig2 = plt.figure(figsize=(10,6))
    # plt.xlabel('Time', fontsize=15)
    # plt.ylabel('Amplitude', fontsize=15)
    # plt.title("Reconstructed Signal")
    # plt.plot(nT , y2)
    # plt.grid(True)
    # st.plotly_chart(fig2)
    signal_ploting_function(f"Reconstructed Signal",nT,y2)


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
        
        # fig1 = plt.figure(figsize=(10,6))
        # plt.xlabel('Time', fontsize=15)
        # plt.ylabel('Amplitude', fontsize=15)
        # plt.title(f"Original {title}")
        # plt.plot(time_data,amplitude_data)
        # plt.plot(nT, y2, 'ro')
        # plt.grid(True)
        # st.plotly_chart(fig1)
        # signal_ploting_function(f"Original {title}",time_data,amplitude_data,nT,y2)
        
        

        # Plotting Reconstructed Signal

        # fig2 = plt.figure(figsize=(10,6))
        # plt.xlabel('Time', fontsize=15)
        # plt.ylabel('Amplitude', fontsize=15)
        # plt.title(f"Reconstructed {title}")
        # plt.plot(nT, y2)
        # plt.grid(True)
        # st.plotly_chart(fig2)
        # signal_ploting_function(f"Reconstructed {title}",nT,y2)

        col1, col2 = st.columns(2,gap='medium')
        
        

        with col1:
         signal_ploting_function(f"Original {title}",time_data,amplitude_data,nT,y2)

        with col2:
            signal_ploting_function(f"Reconstructed {title}",nT,y2)

        
            

        
        
        

    else:
        st.write('Awaiting CSV file to be uploaded.')