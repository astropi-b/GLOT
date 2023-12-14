#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:24:50 2023

@author: agastya
"""

import streamlit as st
from struct import unpack
import pyvisa
import time
import csv
import os
import pandas as pd
from datetime import datetime
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# First, we need to establish a connection with the oscilloscope
rm = pyvisa.ResourceManager()
DATA_DIR = '/Users/agastya/Desktop/rri_test/data'  # Path to your data
LOG_DIR = '/Users/agastya/Desktop/rri_test/log'  # Path to your log directory
#DATA_DIR = '/home/summer/lpda_observation/observation_data'  # Path to your data
#LOG_DIR = '/home/summer/lpda_observation/log'  # Path to your log directory


def freq_to_bin(frequency, segment_size):
    bin_size = 625.0 / (segment_size / 2)  # Bin size in MHz
    bin_number = round(frequency / bin_size)  # Bin number calculation, adding 1 since bins start from 1
    return bin_number


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))

# Define data processing functions
def running_mean(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')

def perform_rfft(voltage, segment_size):
    rfft_results = []
    original_list = voltage
    num_segments = len(original_list) // segment_size
    segments = [original_list[i:i+segment_size] for i in range(0, num_segments*segment_size, segment_size)]
    for segment in segments:
        rfft_results.append(np.fft.rfft(segment)[1:])
    return rfft_results

def perform_rfft_average(voltage, segment_size):
    perform_rfft_average = []
    perform_rfft_average.append(np.mean(np.abs(perform_rfft(voltage, segment_size))**2,axis=0))
    return perform_rfft_average

def correlation(aquisition_1,aquisition_2, segment_size):
    correlated = []
    correlated.append(np.mean(perform_rfft(aquisition_1, segment_size)*np.conj(perform_rfft(aquisition_2, segment_size)),axis=0))
    return correlated

def plot_data(ch_1, ch_2):
    # First subplot: 1, 1
    plt.subplot(2, 2, 1)  # (rows, columns, subplot_number)
    plt.plot(ch_1[1])
    plt.title('Channel 1 Signal')
    plt.xlabel('Time')
    plt.ylabel('Voltage')

    # Second subplot: 1, 2
    plt.subplot(2, 2, 2)
    plt.hist(ch_1[1],50)
    plt.title('Channel 1 Histogram')

    # Third subplot: 2, 1
    plt.subplot(2, 2, 3)
    plt.plot(ch_2[1])
    plt.title('Channel 2 Signal')
    plt.xlabel('Time')
    plt.ylabel('Voltage')

    # Fourth subplot: 2, 2
    plt.subplot(2, 2, 4)
    plt.hist(ch_2[1],50)
    plt.title('Channel 2 Histogram')

    plt.tight_layout()  # Adjusts the spacing between subplots
    st.pyplot(plt)  # This will display the plot in Streamlit


def perform_data_processing(selected_folder, segment_size, files_to_process=None):
    current_dir = os.path.join(DATA_DIR, selected_folder)
    files = os.listdir(current_dir)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(current_dir, x)))

    if files_to_process is not None:
        files = [file for file in files if file in files_to_process]

    rows_to_skip = 1
    rfft_ch1 = []
    rfft_ch2 = []
    correlated = []
    ch_1=[]
    ch_2=[]
    for file_name in files:
        if file_name.endswith('.csv'):
            file_path = os.path.join(current_dir, file_name)
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                for _ in range(rows_to_skip):
                    next(reader)  # Skip rows
                aquisition_1 = []
                aquisition_2 = []
                for row in reader:
                    if len(row)>=1:
                        aquisition_1.append(float(row[0])-127.5)
                        aquisition_2.append(float(row[1])-127.5)
                ch_1.append(aquisition_1)
                ch_2.append(aquisition_2)
                rfft_ch1.append(perform_rfft_average(aquisition_1, segment_size))
                rfft_ch2.append(perform_rfft_average(aquisition_2, segment_size))
                correlated.append(correlation(aquisition_1,aquisition_2, segment_size))
    st.markdown('## Time Series Data Check')
    plot_data(ch_1, ch_2)
    process_and_display_results(rfft_ch1, rfft_ch2, correlated)
    return rfft_ch1,rfft_ch2,correlated

def advance_data_processing(selected_folder, segment_size,x_values,y_values,files_to_process=None):
    current_dir = os.path.join(DATA_DIR, selected_folder)
    files = os.listdir(current_dir)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(current_dir, x)))
    timestamps = [datetime.strptime('_'.join(os.path.splitext(file)[0].split('_')[3:]), '%Y-%m-%d_%H-%M-%S') for file in files]
    timestamps=sorted(timestamps)

    if files_to_process is not None:
        files = [file for file in files if file in files_to_process]

    rows_to_skip = 1
    rfft_ch1 = []
    rfft_ch2 = []
    correlated = []
    ch_1=[]
    ch_2=[]
    for file_name in files:
        if file_name.endswith('.csv'):
            file_path = os.path.join(current_dir, file_name)
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                for _ in range(rows_to_skip):
                    next(reader)  # Skip rows
                aquisition_1 = []
                aquisition_2 = []
                for row in reader:
                    if len(row)>=1:
                        aquisition_1.append(float(row[0])-127.5)
                        aquisition_2.append(float(row[1])-127.5)
                ch_1.append(aquisition_1)
                ch_2.append(aquisition_2)
                rfft_ch1.append(perform_rfft_average(aquisition_1, segment_size))
                rfft_ch2.append(perform_rfft_average(aquisition_2, segment_size))
                correlated.append(correlation(aquisition_1,aquisition_2, segment_size))
                    
    display_results(rfft_ch1, rfft_ch2, correlated,x_values,y_values,timestamps)

def display_results(rfft_ch1, rfft_ch2, correlated,x_values,y_values,timestamps):
    rfft_avg_ch1=np.array(rfft_ch1)[:,0]
    rfft_avg_ch2=np.array(rfft_ch2)[:,0]
    correlated_avg=np.array(correlated)[:,0]
    
    st.write("Data is processing")

    # Create DataFrame for CH1
    df_ch1 = create_dataframe(rfft_avg_ch1, "Average of abs squared FFT values of CH1")
    
    # Create DataFrame for CH2
    df_ch2 = create_dataframe(rfft_avg_ch2, "Average of abs squared FFT values of CH2")

    # Create DataFrame for Correlated
    df_correlated = create_dataframe(correlated_avg, "Average of Correlated Segments of CH1 and CH2")
    
    # Masking for df_ch1_reduced
    df_ch1_reduced = df_ch1.copy()
    for x, y in zip(x_values, y_values):
        df_ch1_reduced.loc[:, f'fbin_{x}':f'fbin_{y}'] = 0

    # Masking for df_ch2_reduced
    df_ch2_reduced = df_ch2.copy()
    for x, y in zip(x_values, y_values):
        df_ch2_reduced.loc[:, f'fbin_{x}':f'fbin_{y}'] = 0

    # Masking for df_correlated_reduced
    df_correlated_reduced = df_correlated.copy()
    for x, y in zip(x_values, y_values):
        df_correlated_reduced.loc[:, f'fbin_{x}':f'fbin_{y}'] = 0

    
    
    
    # Compute Time Series Power Spectrum of Channel 1 and Channel 2
    ch1_r = 10*np.log10(df_ch1_reduced.mean(axis=1))
    ch2_r = 10*np.log10(df_ch2_reduced.mean(axis=1))
    correlated_r= 10*np.log10(np.abs(df_correlated_reduced).mean(axis=1))
    
    plot_masked_regions(df_ch1, x_values, y_values, "df_ch1_reduced")
    plot_masked_regions(df_ch2, x_values, y_values, "df_ch2_reduced")
    plot_masked_regions(np.abs(df_correlated), x_values, y_values, "df_correlated_reduced")
    
    # Create and display time series plots
    create_and_display_time_plot2(ch1_r, "Average Power across all Frequcies With Time - CH1", 'blue',timestamps)
    create_and_display_time_plot2(ch2_r, "Average Power across all Frequcies With Time - CH2", 'blue',timestamps)
    create_and_display_time_plot2(correlated_r,"Average Power across all Frequcies With Time - Corr",'blue',timestamps)
    

def create_and_display_dataframe(data, title):
    df=pd.DataFrame(data)
    df.columns = [f'fbin_{i}' for i in range(1, df.shape[1] + 1)]
    df.index = [f'packet_{i}' for i in range(1, len(data) + 1)]

    # Display DataFrame
    st.markdown(f"## {title}")
    st.dataframe(df)
    return df

def create_dataframe(data, title):
    df=pd.DataFrame(data)
    df.columns = [f'fbin_{i}' for i in range(1, df.shape[1] + 1)]
    df.index = [f'packet_{i}' for i in range(1, len(data) + 1)]
    return df


def create_and_display_plot(data, title, color):
    fig, ax = plt.subplots(figsize=(10,5))

    if isinstance(data, list):
        for idx, d in enumerate(data):
            ax.plot(d, color=color[idx], label=f'Channel {idx + 1}')
        ax.legend()
    else:
        ax.plot(data, color=color)

    ax.grid()
    ax.set_xlabel('Frequency Bins')
    ax.set_ylabel('Average Power')
    ax.set_title(title)

    # Display Plot
    st.markdown(f"## {title}")
    st.pyplot(fig)


def create_and_display_spectrograph(data, title):
    fig, ax = plt.subplots()
    image = ax.imshow(np.abs(data).T, cmap='jet', aspect='auto')
    cbar = plt.colorbar(image)
    ax.set_title(title)
    ax.set_xlabel('Packet Number')
    ax.set_ylabel('Frequency Bins')

    # Display Plot
    st.markdown(f"## {title}")
    st.pyplot(fig)

def create_and_display_waterfall_plot(df, title):
    fig, ax = plt.subplots()
    image = ax.imshow(df.T, cmap='jet', aspect='auto')
    cbar = plt.colorbar(image)
    ax.set_title(title)
    ax.set_xlabel('Packet Number')
    ax.set_ylabel('Frequency Bins')

    # Display Plot
    st.markdown(f"## {title}")
    st.pyplot(fig)

import matplotlib.pyplot as plt
import streamlit as st

def plot_masked_regions(df, x_values, y_values, title_prefix):
    for x, y in zip(x_values, y_values):
        fig, ax = plt.subplots()
        image = ax.imshow(df.loc[:, f'fbin_{x}':f'fbin_{y}'].T, cmap='jet', aspect='auto')
        plt.colorbar(image)
        plt.title(f'Masked Region {x} to {y} for {title_prefix}')
        st.pyplot(fig)  # Display the plot in Streamlit




import matplotlib.pyplot as plt

def zoom_and_display_plot(df, y_start, y_end):
    fig, ax = plt.subplots()
    image = ax.imshow(df.loc[:, f'fbin_{y_start}':f'fbin_{y_end}'].T, cmap='jet', aspect='auto')
    plt.show()
    st.pyplot(fig) # Display the plot in Streamlit


def process_and_display_results(rfft_ch1, rfft_ch2, correlated):
    rfft_avg_ch1=np.array(rfft_ch1)[:,0]
    rfft_avg_ch2=np.array(rfft_ch2)[:,0]
    correlated_avg=np.array(correlated)[:,0]
    
    st.write("Data processing is done")

    # Create DataFrame for CH1
    df_ch1 = create_and_display_dataframe(rfft_avg_ch1, "Average of abs squared FFT values of CH1")
    
    # Create DataFrame for CH2
    df_ch2 = create_and_display_dataframe(rfft_avg_ch2, "Average of abs squared FFT values of CH2")

    # Create DataFrame for Correlated
    df_correlated = create_and_display_dataframe(correlated_avg, "Average of Correlated Segments of CH1 and CH2")
    
    # Compute Power Spectrum of Channel 1
    power_spectrum_ch1 = 10*np.log10(np.mean(rfft_avg_ch1,axis=0))
    power_spectrum_ch2 = 10*np.log10(np.mean(rfft_avg_ch2,axis=0))
    correlated_spectrum=10*np.log(np.mean(np.abs(correlated_avg),axis=0))
    
    # Compute Time Series Power Spectrum of Channel 1 and Channel 2
    time_series_power_ch1 = 10 * np.log10(np.mean(rfft_avg_ch1, axis=1))
    time_series_power_ch2 = 10 * np.log10(np.mean(rfft_avg_ch2, axis=1))
    time_series_power_correlated= 10*np.log(np.mean(np.abs(correlated_avg),axis=1))
    

    # Create and display plots
    create_and_display_plot(power_spectrum_ch1, "Power Spectra of CH1",'red')
    create_and_display_plot(power_spectrum_ch2, "Power Spectra of CH2",'blue')
    create_and_display_plot([power_spectrum_ch1, power_spectrum_ch2], "Power Spectra Comparison", ['red', 'blue'])
    create_and_display_plot(correlated_spectrum, "Correlated Spectrum", 'blue')
    
    # Create and display time series plots
    create_and_display_time_series_plot(time_series_power_ch1, "Average Power across all Frequcies With Time - CH1", 'blue')
    create_and_display_time_series_plot(time_series_power_ch2, "Average Power across all Frequcies With Time - CH2", 'blue')
    create_and_display_time_series_plot(time_series_power_correlated,"Average Power across all Frequcies With Time - Corr",'blue')

    # Create and display waterfall and spectrograph plots
    create_and_display_waterfall_plot(df_ch1, "Spectrogram_CH1")
    create_and_display_waterfall_plot(df_ch2, "Spectrogram_CH2")
    create_and_display_spectrograph(df_correlated, "Spectrogram_Correlated")

def create_and_display_time_series_plot(data, title, color):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(data, color=color)
    ax.plot(running_mean(data,50),color='red')
    ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Power')
    ax.set_title(title)

    # Display Plot
    st.markdown(f"## {title}")
    st.pyplot(fig)
def create_and_display_time_plot(data, title, color,timestamps):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(timestamps,data, color=color)
    #ax.plot(running_mean(data,50),color='red')
    ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Power')
    ax.set_title(title)

    # Display Plot
    st.markdown(f"## {title}")
    st.pyplot(fig)
    
def create_and_display_time_plot2(data, title, color, timestamps):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(timestamps, data, color=color)
    
    # Compute running mean
    running_mean_data = running_mean(data, 50)
    
    # Adjust timestamps for the running mean
    adjusted_timestamps = timestamps[len(timestamps) - len(running_mean_data):]
    
    ax.plot(adjusted_timestamps, running_mean_data, color='red')
    ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Power')
    ax.set_title(title)

    # Display Plot
    st.markdown(f"## {title}")
    st.pyplot(fig)

def collect_channel_data(channel, scope):
    scope.write(f'DATA:SOU {channel}')
    scope.write('DATA:WIDTH 1')
    scope.write('DATA:ENC RPB')
    scope.write('CURVE?')
    data = scope.read_raw()

    headerlen = 2 + int(data[1])  # Finding header length
    header = data[:headerlen]  # Separating header
    ADC_wave = data[headerlen:-1]  # Separating data

    ADC_wave = np.array(unpack('%sB' % len(ADC_wave),ADC_wave))
    return ADC_wave

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the task you want to do", ["Welcome", "Data Acquisition", "Data Processing","Data Center"])
    if app_mode == "Welcome":
        st.title("Welcome to the GBD Remote Data Collection and Analysis System")
        st.text("Please navigate to the task you want to perform using the sidebar.")
        st.markdown("This Website helps you with collecting Data from Gauribidanur Radio Observatory where 8 Log Periodic Dipole Antennas are seted up as 2 Element system and each element is connected as a channel to the oscilloscope, and here you can also do some preliminary Data Analysis.")   
        #st.image("/home/summer/lpda_observation/8.png")  # 
        st.image("//Users/agastya/Desktop/rri_test/p.png")
    elif app_mode == "Data Acquisition":
        st.subheader("LPDA Data Acquisition")
        user_name = st.text_input("Enter your name: ")
        
        if user_name:
            # Ask user for inputs
            ip_address = st.text_input("Enter the IP address of the oscilloscope: ")
            channels_input = st.text_input("Enter the active channels separated by comma (e.g., CH1,CH3): ")
            start_time_input = st.text_input("Enter the start time (in format YYYY-MM-DD HH-MM): ")
            end_time_input = st.text_input("Enter the end time (in format YYYY-MM-DD HH-MM): ")
            delay = st.number_input("Enter the delay time in seconds between each data collection(min 2 seconds ): ", value=2.0, step=0.5)
            folder_name = st.text_input("Enter the name of the folder to save the files: ", value='')

            if st.button("Start data acquisition"):
                
                # Log user details and their action
                log_file = os.path.join(LOG_DIR, 'user_actions.csv')
                with open(log_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([user_name, datetime.now(), ip_address,channels_input, start_time_input, end_time_input, delay, folder_name])

                # Create the folder if it doesn't exist
                folder_path = os.path.join(DATA_DIR, folder_name)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                # rest of your data acquisition code
                rm = pyvisa.ResourceManager()
                scope = rm.open_resource(f'TCPIP::{ip_address}::INSTR')
                
                settings = scope.query('*LRN?') 
                current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                filename = f'oscilloscope_settings_{current_datetime}.txt'

                # Create the full file path
                file_path = os.path.join(LOG_DIR, filename)

                # Save the settings to the specified path
                with open(file_path, 'w') as file:
                    file.write(settings)

                # Convert user inputs to datetime objects and channels to a list
                try:
                    start_time = datetime.strptime(start_time_input, "%Y-%m-%d %H-%M")
                    end_time = datetime.strptime(end_time_input, "%Y-%m-%d %H-%M")
                except ValueError:
                    st.error("Invalid time format. Please enter the time in the format YYYY-MM-DD HH-MM")
                    return
                channels = [channel.strip() for channel in channels_input.split(',')]

                # Counter for packets
                packet_counter = 1 

                # Get current time
                now = datetime.now()

                while now < end_time:
                    # Wait until start time
                    if now < start_time:
                        time.sleep(3)  # wait for 3 seconds before checking again
                        now = datetime.now()
                        continue

                    # Trigger a single acquisition
                    scope.write('ACQUIRE:STATE OFF')

                    # Collect the data from the channels
                    data_channels = [collect_channel_data(channel, scope) for channel in channels]

                    # Create a new CSV for this timestamp
                    timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')  # current time
                    filename = os.path.join(folder_path, f'oscilloscope_data_{packet_counter}_{timestamp}.csv')

                    with open(filename, 'w', newline='') as file:
                        writer = csv.writer(file)
                        headers = [f"Packet{packet_counter} {channel}" for channel in channels] 
                        writer.writerow(headers)  # CSV header
                        for data_row in zip(*data_channels):
                            writer.writerow(list(data_row))
                          
                    
                    scope.write('ACQUIRE:STATE ON')
                    time.sleep(delay)

                    # Increment packet counter
                    packet_counter += 1
                    now = datetime.now()

                # Remember to close the connection
                scope.close()
                st.write("Data collection is done. You can see the data in the Data Center")

             

    if app_mode == "Data Processing":
        st.subheader("Initial Data Processing")

    # Ask user for inputs
        selected_folder = st.selectbox("Select a folder to process the data:", os.listdir(DATA_DIR))
        
        
        

    # Ask the user to choose between full data processing and quick check
        data_processing_mode = st.selectbox("Choose the data processing mode:", ['Full Data', 'Partial Data Anaylsis'])

        if data_processing_mode == 'Full Data':
            segment_size = st.selectbox("Choose the segment size:", [256, 512, 1024, 2048])
            start_processing = st.button("Start data processing")

            if start_processing:
                # Perform data processing
                #perform_data_processing(selected_folder, segment_size)
                perform_data_processing(selected_folder, segment_size)
                
              


        elif data_processing_mode == 'Partial Data Anaylsis':
            current_dir = os.path.join(DATA_DIR, selected_folder)
            files = os.listdir(current_dir)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(current_dir, x)))

            first_file = st.selectbox("Select the first file for processing:", files)
            second_file = st.selectbox("Select the second file for processing:", files)
            segment_size = st.selectbox("Choose the segment size:", [256, 512, 1024, 2048])
            start_processing = st.button("Start data processing")

            if start_processing:
            # Perform data processing between the two selected files
                file_start = files.index(first_file)
                file_end = files.index(second_file)
                files_to_process = files[file_start: file_end + 1]
            
            # Perform data processing but only for files_to_process
                perform_data_processing(selected_folder, segment_size, files_to_process)
        
     
         # Additional code to handle zooming
       
        st.subheader("Data Masking and Final Results")
        
        selected_folder2 = st.selectbox("Select a folder to process the data:", os.listdir(DATA_DIR), key="older2")
                        
        
       # Ask the user to choose between full data processing and quick check
        data_processing_mode2 = st.selectbox("Choose the data processing mode:", ['Full Data'],key="unique")

        if data_processing_mode2 == 'Full Data':
            segment_size2 = st.selectbox("Choose the segment size:", [256, 512, 1024, 2048],key="uniqu")
            number_of_regions = st.number_input("How many regions do you want to mask?", min_value=0, value=0, step=1)
            x_values = []
            y_values = []
            regions = []
        
            for i in range(number_of_regions):
                st.markdown(f"### Region {i + 1}")
                x = st.number_input(f"Enter the intial frequency bin value for region {i + 1}:", value=0)
                y = st.number_input(f"Enter the final frequency bin value for region {i + 1}:", value=0)
                x_values.append(x)
                y_values.append(y)

            if st.button("Confirm Regions"):
                for x, y in zip(x_values, y_values):
                    regions.append((x, y))
                st.write("You have marked the following regions for masking:")
                for i, region in enumerate(regions, 1):
                    st.write(f"Region {i}: {region}")
                    
            start_processing2 = st.button("Start data processing",key="unque")
            

            if start_processing2:
                # Perform data processing
                #perform_data_processing(selected_folder, segment_size)
                advance_data_processing(selected_folder2, segment_size2,x_values,y_values)
                

        
           
        
        
    elif app_mode == "Data Center":
        st.subheader("LPDA ")
        # Get all the folders in the directory
        data_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]

        if data_folders:
            # Let the user select the folder
            selected_folder = st.selectbox("Select a folder to view its files", data_folders)

            # Get all the csv files in the selected folder
            data_files = [f for f in os.listdir(os.path.join(DATA_DIR, selected_folder)) if f.endswith('.csv')]

            if data_files:
                # Let the user select the file
                file_to_load = st.selectbox("Select a file to view", data_files)

                if st.button("Load Data"):
                    df = pd.read_csv(os.path.join(DATA_DIR, selected_folder, file_to_load))
                    st.write(df)
                    st.success("Data Loaded")

                    # Download Button for Single File
                    st.download_button(
                        label="Download CSV File",
                        data=df.to_csv(index=False),
                        file_name=f"{file_to_load}",
                        mime="text/csv",
                    )

                # Zip and Download Button for Whole Folder
                with ZipFile('temp.zip', 'w') as zipf:
                    zipdir(os.path.join(DATA_DIR, selected_folder), zipf)
                
                with open('temp.zip', 'rb') as f:
                    bytes = f.read()

                st.download_button(
                    label="Download Zip of Folder",
                    data=bytes,
                    file_name=f"{selected_folder}.zip",
                    mime="application/zip",
                )
                
                os.remove('temp.zip')  # remove temporary file

            else:
                st.info("No data files available in the selected folder.")
        else:
            st.info("No folders available.")
    
        
        
       
    st.sidebar.markdown('---')        
    # Use the sidebar for inputs
    st.sidebar.header('Calculator')

    # Create a checkbox for user to choose if they want to provide frequency input
    freq_input_check = st.sidebar.checkbox('Frequency Bin Calculator')

    # If checkbox is checked, display input fields
    if freq_input_check:
        segment_size = st.sidebar.selectbox('Segment size', options=[256, 512, 1024, 2048])
        frequency1 = st.sidebar.number_input('Frequency 1 (in MHz)', min_value=0.0, max_value=625.0, value=0.0)
        frequency2 = st.sidebar.number_input('Frequency 2 (in MHz)', min_value=0.0, max_value=625.0, value=0.0)
    
        # Create a button for user to calculate results
        calculate_button = st.sidebar.button('Calculate Bin Numbers')
    
        # If calculate button is pressed, calculate and display bin numbers
        if calculate_button:
            bin_number1 = freq_to_bin(frequency1, segment_size)
            bin_number2 = freq_to_bin(frequency2, segment_size)
    
             # Display bin numbers in the sidebar
            st.sidebar.write(f'Bin number for Frequency 1 ({frequency1} MHz): {bin_number1}')
            st.sidebar.write(f'Bin number for Frequency 2 ({frequency2} MHz): {bin_number2}')
    # Display developers' names at the bottom
    st.sidebar.markdown('---')
    st.sidebar.markdown('**Developers:**')
    st.sidebar.markdown('A.A.S.LIKHIT')
    st.sidebar.markdown('astropi.2003@gmail.com')
    st.sidebar.markdown('Katta Naveen')
    st.sidebar.markdown('naveenkatta626@gmail.com')
    
     


if __name__ == "__main__":
    main()