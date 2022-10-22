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


st.set_page_config(layout='wide')


with open('app.css') as fileStyle:
    st.markdown(f'<style>{fileStyle.read()}</style>', unsafe_allow_html=True)
# Functions

# Plotting


def simple_plot(signal='', x1=[], y1=[]):
    fig = plt.figure(figsize=(1, 5))
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Amplitude', fontsize=15)
    plt.title(signal, fontsize=25)
    plt.plot(x1, y1)
    plt.grid(True)
    st.plotly_chart(fig, use_container_width=True)


def plot(signal='', x1=[], y1=[], x2=[], y2=[], x3=[], y3=[], interp=False):

    fig = plt.figure(figsize=(1, 5))
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Amplitude', fontsize=15)
    plt.title(signal, fontsize=25)
    plt.plot(x1, y1)
    plt.grid(True)

    if len(x2) != 0 and len(y2) != 0:
        plt.plot(x2, y2, 'ro')  # Sampling points

    if (s_Interpolation):
        plt.plot(x3, y3, '--')  # Sampling interpulation

    st.plotly_chart(fig, use_container_width=True)

# Summation


def summation_sins(amplitude, frequency, time_axis):
    n = len(frequency)
    sinewave = np.zeros(len(time_axis))
    for i in range(n):
        sinewave += amplitude[i] * np.sin(2 * np.pi * frequency[i] * time_axis)

    return sinewave


# Noise

def Noise_using_snr(snr, signal):

    sigpower = sum([math.pow(abs(signal[i]), 2) for i in range(len(signal))])
    sigpower = sigpower/len(signal)
    noisepower = sigpower/(math.pow(10, snr/10))
    noise = math.sqrt(noisepower)*(np.random.uniform(-1, 1, size=len(signal)))
    return noise


# Sidebar
menu = st.sidebar.radio('menu', options=['Main', 'Generation', 'CSV'])

if menu == 'Main':
    st.title('Sampling Studio')
    st.text('This is a web app to show sampled signals')


#  Generation Page

elif menu == 'Generation':
    st.title('Generation')

    gencol1, gencol2 = st.columns([1, 3])

    with gencol1:

        # Summation

        # addcol1, addcol2, addcol3 = st.columns(3)
        X = st.number_input("Frequency", step=1)
        Y = st.number_input("Amplitude", step=1)
        signal_name = st.text_input("Signal name", value="Signal_name")

        if st.button("Add Signal"):
            if X > 0 and Y > 0 and signal_name != "":
                Data = [X, Y, signal_name]
                if os.path.exists("DataFile.csv"):
                    with open('DataFile.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(Data)

                else:
                    with open('DataFile.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(['frequency', "Amplitude", "id"])
                        writer.writerow(Data)

        # Deletion
        # remove_specific_row_from_csv(df, "id", id_signal)
        signal_names = pd.read_csv("DataFile.csv").iloc[:, 2]
        added_signal = st.selectbox(
            'select signal you want to delete', (signal_names))
        if st.button("delete signal"):
            df = pd.read_csv("DataFile.csv")
            df = df[df.id != added_signal]
            df.to_csv("DataFile.csv", index=False)

        # delete all signal
        # if st.button("delete All Signals"):
        #     if os.path.exists("DataFile.csv"):
        #         os.remove("DataFile.csv")
        # else:
        #     st.write("The file does not exist")

        # Sample = st.checkbox("Show Sampling")
        if os.path.exists("DataFile.csv"):
            noiseCheck = st.checkbox("Add Noise")
            if (noiseCheck):
                snr = st.slider('SNR', min_value=1,
                                max_value=50, step=1, value=50)
            Sample = st.checkbox("Show Sampling")
            if (Sample):
                sample_freq = st.slider(
                    'Sampling Frequency', min_value=1, max_value=100, step=1)
                s_Interpolation = st.checkbox('Show Interpolation')

    with gencol2:

        if os.path.exists("DataFile.csv"):
            if (not Sample):
                DataTowCo = pd.read_csv("DataFile.csv")
                frequency = DataTowCo.iloc[:, 0]
                Amplitude = DataTowCo.iloc[:, 1]
                t = np.arange(0, 1, 0.001)
                y1 = summation_sins(Amplitude, frequency, t)

                # Noise
                # noiseCheck = st.checkbox("Add Noise")
                if (noiseCheck):
                    # snr = st.slider('SNR', min_value=1,
                    #                 max_value=50, step=1, value=50)
                    noise = Noise_using_snr(snr, y1)
                    y1 = y1 + noise

                simple_plot(f"Generated Signal", t, y1)

            if (Sample):
                DataTowCo = pd.read_csv("DataFile.csv")
                frequency = DataTowCo.iloc[:, 0]
                Amplitude = DataTowCo.iloc[:, 1]

                t = np.arange(0, 1, 0.001)
                y1 = summation_sins(Amplitude, frequency, t)

                s_rate = sample_freq  # Sampling Frequency

                Ts = 1 / s_rate  # How much time for a full cycle

                # To spread the samples right on the graph
                nT = np.arange(0, 1, Ts)

                y2 = summation_sins(Amplitude, frequency, nT)  # Resampled

                if (noiseCheck):
                    noise1 = Noise_using_snr(snr, y1)
                    noise2 = Noise_using_snr(snr, y2)
                    y1 = y1 + noise1
                    y2 = y2 + noise2

                # Sinc Interpolation

                y_reconstruction = np.zeros(len(t))
                for i in range(1, len(t)):
                    for n in range(1, len(nT)):
                        y_reconstruction[i] += y2[n] * np.sinc((t[i]-nT[n])/Ts)

                # Plotting

                plot(f"Original Signal", t, y1, nT, y2, t, y_reconstruction,
                     s_Interpolation)     # Plotting Original Signal
                # Plotting Reconstructed Signal
                simple_plot(f"Reconstructed Signal", t, y_reconstruction)


# CSV Page

elif menu == 'CSV':

    st.title("Upload Signal")

    csvCol1, csvCol2 = st.columns([1, 3])

    with csvCol1:

        # Upload CSV

        uploaded_file = st.file_uploader(
            "Upload your input CSV file", type={"csv", "txt ,xlsx"})

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

            sample_freq = st.slider(
                'Sampling Frequency', min_value=1, max_value=100, step=1)

            s_Interpolation = st.checkbox('Show Interpolation')
            noiseCheck = st.checkbox("Add Noise")
            if (noiseCheck):
                snr = st.slider('SNR', min_value=1,
                                max_value=50, step=1, value=5)
                # noise = Noise_using_snr(snr, amplitude_data)
                # y1 = amplitude_data + noise

                # simple_plot(f"Generated Signal", time_data, y1)
            s_rate = sample_freq  # sampling frequency
            T = 1 / s_rate

            numberOfSamples = (time_maximum - time_minimum)/T

            calc = ceil(numberOfRecords / numberOfSamples)

            nT = time_data[0:numberOfRecords:calc]  # Spreading Samples
            y2 = amplitude_data[0:numberOfRecords:calc]

            # Sinc Interpolation
            y_reconstruction = np.zeros(len(time_data))
            for i in range(0, len(time_data)-1):
                for x, y in zip(nT, y2):
                    y_reconstruction[i] += y * np.sinc((time_data[i]-x)/T)

        else:
            st.write('Awaiting CSV file to be uploaded.')

    with csvCol2:

        if uploaded_file is not None:

            # Plotting

            if (noiseCheck):
                noise1 = Noise_using_snr(snr, amplitude_data)
                amplitude_data = amplitude_data + noise1

            plot(f"Original Signal", time_data, amplitude_data, nT, y2, time_data,
                 y_reconstruction, s_Interpolation)  # Plotting Original Signal
            # Plotting Reconstructed Signal
            simple_plot(f"Reconstructed Signal", time_data, y_reconstruction)
