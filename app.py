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
# with open('app.css') as fileStyle:
#     st.markdown(f'<style>{fileStyle.read()}</style>', unsafe_allow_html=True)

# Functions
# Convert data frame to csv


@st.cache
def convert_df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    df = df.to_csv(index=False)
    return df.encode('utf-8')

# Plotting


def plot(signal='', time=[], value=[], sampleTime=[], sampleValue=[], value_rec=[], hide_original=False, sampling=False, interp=False, x_axis='Time (s)', y_axis='Amplitude'):
    fig = plt.figure(figsize=(1, 5))
    plt.xlabel(x_axis, fontsize=17)
    plt.ylabel(y_axis, fontsize=17)
    plt.title(signal, fontsize=25)
    if(not hide_original):
        plt.plot(time, value)  # Plot oriniganl function

    if (sampling):
        plt.plot(sampleTime, sampleValue, 'ro')  # Sampling points

    if (interp and not hide_original):
        plt.plot(time, value_rec, '--')  # Sampling interpulation

    if (interp and hide_original):
        plt.plot(time, value_rec)  # Sampling interpulation

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

    sigpower = sum([math.pow(abs(signal_value[i]), 2)
                   for i in range(len(signal_value))])
    sigpower = sigpower/len(signal_value)
    noisepower = sigpower/(math.pow(10, snr/10))
    noise = math.sqrt(noisepower) * \
        (np.random.uniform(-1, 1, size=len(signal_value)))
    return noise


# Sidebar

menu = st.sidebar.radio(
    'Menu', options=['Instructions', 'Generate Signal', 'Upload Signal'])

# Instructions Page

if menu == 'Instructions':
    st.title('Sampling Studio')
    st.write(
        '- This is a web app to show signals and sample them with different frequencies.')
    st.write("- From the side bar, you can choose to either generate your own signal of sine waves or upload a signal from a csv file.")
    st.write(
        "- Then you can sample the signal, add noise and see the reconstructed signal.")


#  Generation Page

elif menu == 'Generate Signal':
    st.title('Generate Signal')

    gencol1, space, gencol2 = st.columns([1, 0.2, 4])

    with gencol1:

        # Summation of inputs

        frequency_input = st.number_input("Frequency (Hz)", step=1)
        amplitude_input = st.number_input("Amplitude", step=1)
        signal_name = st.text_input("Signal Name", value="Signal_name")

        if st.button("Add Signal"):
            if frequency_input > 0 and amplitude_input != 0 and signal_name != "":
                Data = [frequency_input, amplitude_input, signal_name]
                if os.path.exists("DataFile.csv"):
                    with open('DataFile.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(Data)

                else:
                    with open('DataFile.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            ['frequency', "Amplitude", "Signal_name"])
                        writer.writerow(Data)

        # Deletion
        # remove_specific_row_from_csv(df, "id", id_signal)

        signal_names = pd.read_csv("DataFile.csv").iloc[:, 2]
        added_signal = st.selectbox(
            'select signal you want to delete', (signal_names))

        if st.button("Delete Signal"):
            df = pd.read_csv("DataFile.csv")
            df = df[df.Signal_name != added_signal]
            df.to_csv("DataFile.csv", index=False)

        # Noise and Sampling Checkbox

        snr = st.slider('SNR', min_value=1, max_value=50, step=1, value=50)
        sample_freq = st.slider('Sampling Frequency (Hz)',
                                min_value=1, max_value=100, step=1)
        sampling = st.checkbox('Show Sample Points')  # Sampling CheckBox
        s_Interpolation = st.checkbox('Show Interpolation')
        hide_original = st.checkbox('Hide Original Signal')

    with gencol2:

        # Plotting Signals

        if os.path.exists("DataFile.csv"):

            # Plotting with sampling

            DataTowCo = pd.read_csv("DataFile.csv")
            frequency = DataTowCo.iloc[:, 0]
            Amplitude = DataTowCo.iloc[:, 1]

            # Original Attributes

            time = np.arange(0, 1, 0.001)
            y_signal = summation_sins(Amplitude, frequency, time)

            # Sampling Attributes

            sample_rate = sample_freq  # Sampling Frequency
            sample_periodic_time = 1 / sample_rate  # How much time for a full cycle
            # To spread the samples right on the graph
            time_samples = np.arange(0, 1, sample_periodic_time)
            y_samples = summation_sins(
                Amplitude, frequency, time_samples)  # Resampled

            # Noise Addittion
            noise1 = Noise_using_snr(snr, y_signal)
            noise2 = Noise_using_snr(snr, y_samples)
            y_signal = y_signal + noise1
            y_samples = y_samples + noise2

            # Sinc Interpolation
            y_reconstruction = np.zeros(len(time))
            for i in range(0, len(time)):
                for n in range(0, len(time_samples)):
                    y_reconstruction[i] += y_samples[n] * \
                        np.sinc((time[i]-time_samples[n])/sample_periodic_time)

            # Plotting Original Signal, Samples and Interpolation
            plot(f"Signal", time, y_signal, time_samples, y_samples, y_reconstruction,
                 hide_original, sampling, s_Interpolation)     # Plotting Original Signal


# CSV Page

elif menu == 'Upload Signal':

    st.title("Upload Signal")

    csvCol1, space, csvCol2 = st.columns([1, 0.1, 2])

    with csvCol1:

        # Upload CSV

        uploaded_file = st.file_uploader("", type={"csv"})

        if uploaded_file is not None:

            file_name = uploaded_file.name
            file_name = file_name[0:-4]
            input_df = pd.read_csv(uploaded_file)
            df = input_df.dropna(axis=0, how='any')

            x_axis = input_df.iloc[0:0, 0].name
            y_axis = input_df.iloc[0:0, 1].name

            # Read data

            time_data = df[df.head(0).columns[0]]
            amplitude_data = df[df.head(0).columns[1]]
            time_maximum = time_data.max()
            time_minimum = time_data.min()
            numberOfRecords = len(time_data)

            # Sample Attributes
            snr = st.slider('SNR', min_value=1, max_value=50, step=1, value=50)
            # Sampling Freq Slider
            sample_freq = st.slider(
                'Sampling Frequency (Hz)', min_value=1, max_value=100, step=1)
            sampling = st.checkbox('Show Sample Points')  # Sampling CheckBox
            s_Interpolation = st.checkbox(
                'Show Interpolation')  # Interpolation CheckBox
            hide_original = st.checkbox('Hide Original Signal')

            sample_rate = sample_freq  # sampling frequency
            sample_periodic_time = 1 / sample_rate

            numberOfSamples = (time_maximum - time_minimum) / \
                sample_periodic_time
            steps = ceil(numberOfRecords / numberOfSamples)
            # Spreading Samples
            time_samples = time_data[0:numberOfRecords:steps]
            y_samples = amplitude_data[0:numberOfRecords:steps].to_numpy()

            noise1 = Noise_using_snr(snr, amplitude_data)
            noise2 = Noise_using_snr(snr, y_samples)
            amplitude_data = amplitude_data + noise1
            y_samples = y_samples + noise2

            # Sinc Interpolation
            y_reconstruction = np.zeros(len(time_data))
            for i in range(0, len(time_data)):
                for x, y in zip(time_samples, y_samples):
                    y_reconstruction[i] += y * \
                        np.sinc((time_data[i]-x)/sample_periodic_time)

        else:
            st.write('Awaiting CSV file to be uploaded.')

    with csvCol2:

        if uploaded_file is not None:
            # Plotting Original Signal, Samples and Interpolation
            plot(file_name, time_data, amplitude_data, time_samples, y_samples, y_reconstruction,
                 hide_original, sampling, s_Interpolation, x_axis, y_axis)  # Plotting Original Signal
