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
import math
from traitlets import default

# Page Layout

st.set_page_config(layout='wide')
with open('app.css') as fileStyle:
    st.markdown(f'<style>{fileStyle.read()}</style>', unsafe_allow_html=True)

# Functions
# Convert data frame to csv

@st.cache
def convert_df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    df = df.to_csv(index=False)
    return df.encode('utf-8')


# Plotting 1 function

def simple_plot(signal='', time=[], value=[]):
    fig = plt.figure(figsize=(1, 5))
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Amplitude', fontsize=15)
    plt.title(signal, fontsize=25)
    plt.plot(time, value)
    plt.grid(True)
    st.plotly_chart(fig, use_container_width=True)

# Plotting multiple functions

def plot(signal='', time=[], value=[], sampleTime=[], sampleValue=[], time_rec=[], value_rec=[], interp=False):
    fig = plt.figure(figsize=(1, 5))
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Amplitude', fontsize=15)
    plt.title(signal, fontsize=25)
    plt.plot(time, value) # Plot oriniganl function
    plt.grid(True)

    if len(sampleTime) != 0 and len(sampleValue) != 0:
        plt.plot(sampleTime, sampleValue, 'ro')  # Sampling points

    if (interp):
        plt.plot(time_rec, value_rec, '--')  # Sampling interpulation

    st.plotly_chart(fig, use_container_width=True)

# Summation of multiple sinewaves

def summation_sins(amplitude, frequency, time_axis):
    number_of_records = len(frequency)
    sinewave = np.zeros(len(time_axis))
    for i in range(number_of_records):
        sinewave += amplitude[i] * np.sin(2 * np.pi * frequency[i] * time_axis)

    return sinewave


# Noise generation

def Noise_using_snr(snr, signal_value):

    sigpower = sum([math.pow(abs(signal_value[i]), 2) for i in range(len(signal_value))])
    sigpower = sigpower/len(signal_value)
    noisepower = sigpower/(math.pow(10, snr/10))
    noise = math.sqrt(noisepower)*(np.random.uniform(-1, 1, size=len(signal_value)))
    return noise


# Sidebar

menu = st.sidebar.radio('Menu', options=['Instructions', 'Generate Signal', 'Upload Signal'])

# Instructions Page

if menu == 'Instructions':
    st.title('Sampling Studio')
    st.write('- This is a web app to show signals and sample them with different frequences.')
    st.write("- From the side bar, you can choose to either generate your own signal or upload a signal from a csv file.")
    st.write("- Then you can sample the signal, add noise and see the reconstructed signal.")


#  Generation Page

elif menu == 'Generate Signal':
    st.title('Generate Signal')

    gencol1, gencol2 = st.columns([1, 3])

    with gencol1:

        # Summation of inputs

        frequency_input = st.number_input("Frequency", step=1)
        amplitude_input = st.number_input("Amplitude", step=1)
        signal_name = st.text_input("Signal Name", value="Signal_name")

        if st.button("Add Signal"):
            if frequency_input > 0 and signal_name != "":
                Data = [frequency_input, amplitude_input, signal_name]
                if os.path.exists("DataFile.csv"):
                    with open('DataFile.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(Data)

                else:
                    with open('DataFile.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(['frequency', "Amplitude", "Signal_name"])
                        writer.writerow(Data)

        # Deletion
        # remove_specific_row_from_csv(df, "id", id_signal)

        signal_names = pd.read_csv("DataFile.csv").iloc[:, 2]
        added_signal = st.selectbox('select signal you want to delete', (signal_names))

        if st.button("Delete Signal"):
            df = pd.read_csv("DataFile.csv")
            df = df[df.Signal_name != added_signal]
            df.to_csv("DataFile.csv", index=False)

        # Download Signal

        frequency = pd.read_csv("DataFile.csv").iloc[:, 0] 
        Amplitude = pd.read_csv("DataFile.csv").iloc[:, 1]
        time = np.arange(0, 1, 0.001)
        y_signal = summation_sins(Amplitude, frequency, time)
        dataset = pd.DataFrame({'X': time, 'Y': y_signal})
        st.download_button(
            label="Download data as CSV",
            data=convert_df_to_csv(dataset),
            file_name='selected_df.csv')

        # Noise and Sampling Checkbox

        if os.path.exists("DataFile.csv"):
            noiseCheck = st.checkbox("Add Noise")
            if (noiseCheck):
                snr = st.slider('SNR', min_value=1, max_value=50, step=1, value=50)
            Sample = st.checkbox("Show Sampling")
            if (Sample):
                sample_freq = st.slider('Sampling Frequency', min_value=1, max_value=100, step=1)
                s_Interpolation = st.checkbox('Show Interpolation')

    with gencol2:

        # Plotting Signals

        if os.path.exists("DataFile.csv"):
            if (not Sample):
                DataTowCo = pd.read_csv("DataFile.csv")
                frequency = DataTowCo.iloc[:, 0]
                Amplitude = DataTowCo.iloc[:, 1]
                time = np.arange(0, 1, 0.001)
                y_signal = summation_sins(Amplitude, frequency, time)
                dataset = pd.DataFrame({'X': time, 'Y': y_signal})
                # Noise

                if (noiseCheck):
                    noise = Noise_using_snr(snr, y_signal)
                    y_signal = y_signal + noise

                simple_plot(f"Generated Signal", time, y_signal)

            # Plotting with sampling

            if (Sample):
                DataTowCo = pd.read_csv("DataFile.csv")
                frequency = DataTowCo.iloc[:, 0]
                Amplitude = DataTowCo.iloc[:, 1]

                # Original Attributes

                time = np.arange(0, 1, 0.001)
                y_signal = summation_sins(Amplitude, frequency, time)

                # Sampling Attributes

                sample_rate = sample_freq  # Sampling Frequency
                sample_periodic_time = 1 / sample_rate  # How much time for a full cycle
                time_samples = np.arange(0, 1, sample_periodic_time) # To spread the samples right on the graph
                y_samples = summation_sins(Amplitude, frequency, time_samples)  # Resampled


                # Noise Addittion
                if (noiseCheck):
                    noise1 = Noise_using_snr(snr, y_signal)
                    noise2 = Noise_using_snr(snr, y_samples)
                    y_signal = y_signal + noise1
                    y_samples = y_samples + noise2

                # Sinc Interpolation
                y_reconstruction = np.zeros(len(time))
                for i in range(0, len(time)):
                    for n in range(0, len(time_samples)):
                        y_reconstruction[i] += y_samples[n] * np.sinc((time[i]-time_samples[n])/sample_periodic_time)

                # Plotting Original Signal, Samples and Interpolation
                plot(f"Original Signal", time, y_signal, time_samples, y_samples, time, y_reconstruction, s_Interpolation)     # Plotting Original Signal
                
                # Plotting Reconstructed Signal
                simple_plot(f"Reconstructed Signal", time, y_reconstruction)


# CSV Page

elif menu == 'Upload Signal':

    st.title("Upload Signal")

    csvCol1, csvCol2 = st.columns([1, 2])

    with csvCol1:

        # Upload CSV

        uploaded_file = st.file_uploader(
            "Upload your input CSV file", type={"csv"})

        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            df = input_df.dropna(axis=0, how='any')
            file_name = uploaded_file.name

            # Read data

            time_data = df[df.head(0).columns[0]]
            amplitude_data = df[df.head(0).columns[1]]
            time_maximum = time_data.max()
            time_minimum = time_data.min()
            numberOfRecords = len(time_data)

            # Sample Attributes

            sample_freq = st.slider('Sampling Frequency', min_value=1, max_value=100, step=1) # Sampling Freq Slider
            s_Interpolation = st.checkbox('Show Interpolation')  # Interpolation CheckBox
            sample_rate = sample_freq  # sampling frequency
            sample_periodic_time = 1 / sample_rate

            numberOfSamples = (time_maximum - time_minimum)/sample_periodic_time
            steps = ceil(numberOfRecords / numberOfSamples)
            time_samples = time_data[0:numberOfRecords:steps]  # Spreading Samples
            y_samples = amplitude_data[0:numberOfRecords:steps].to_numpy()


            # Adding Noise

            noiseCheck = st.checkbox("Add Noise")
            if (noiseCheck):
                snr = st.slider('SNR', min_value=1, max_value=50, step=1, value=50)
                noise1 = Noise_using_snr(snr, amplitude_data)
                noise2 = Noise_using_snr(snr, y_samples)
                amplitude_data = amplitude_data + noise1
                y_samples = y_samples + noise2

            # Sinc Interpolation
            y_reconstruction = np.zeros(len(time_data))
            for i in range(0, len(time_data)-1):
                for x, y in zip(time_samples, y_samples):
                    y_reconstruction[i] += y * np.sinc((time_data[i]-x)/sample_periodic_time)

        else:
            st.write('Awaiting CSV file to be uploaded.')

    with csvCol2:

        if uploaded_file is not None:
            # Plotting Original Signal, Samples and Interpolation
            plot(f"Original Signal", time_data, amplitude_data, time_samples, y_samples, time_data, y_reconstruction, s_Interpolation)  # Plotting Original Signal
        
            # Plotting Reconstructed Signal
            simple_plot(f"Reconstructed Signal", time_data, y_reconstruction)
