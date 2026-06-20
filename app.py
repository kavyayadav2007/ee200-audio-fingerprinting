import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter
import io
import os

# -------------------------------------------------------------------
# 1. SIGNAL PROCESSING & FINGERPRINTING LOGIC
# -------------------------------------------------------------------

def get_spectrogram(audio_data, fs, nperseg=1024):
    """Computes the spectrogram of the audio signal."""
    # Using scipy's spectrogram. nperseg determines the time/freq resolution tradeoff.
    f, t, Sxx = signal.spectrogram(audio_data, fs, window='hann', nperseg=nperseg, noverlap=nperseg//2)
    # Convert to dB scale for better dynamic range handling
    Sxx_db = 10 * np.log10(Sxx + 1e-10) 
    return f, t, Sxx_db

def extract_peaks(Sxx_db, filter_size=20, threshold_db=10):
    """Finds local maxima (the constellation) in the spectrogram."""
    # Apply a maximum filter to find local peaks
    local_max = maximum_filter(Sxx_db, size=filter_size)
    # A peak is where the original spectrogram equals the max filter AND is above a noise threshold
    peak_mask = (Sxx_db == local_max) & (Sxx_db > threshold_db)
    
    # Get the row (freq) and col (time) indices of the peaks
    freq_idx, time_idx = np.where(peak_mask)
    return list(zip(freq_idx, time_idx))

def generate_hashes(peaks, fan_out=5):
    """Pairs nearby peaks to create robust hashes."""
    # Sort peaks by time
    peaks = sorted(peaks, key=lambda x: x[1])
    hashes = []
    
    for i in range(len(peaks)):
        for j in range(1, fan_out + 1):
            if i + j < len(peaks):
                freq1, time1 = peaks[i]
                freq2, time2 = peaks[i + j]
                time_delta = time2 - time1
                
                # The hash is a tuple: (f1, f2, delta_t)
                # We also store time1 to know WHEN this hash occurred in the clip
                hash_tuple = (freq1, freq2, time_delta)
                hashes.append((hash_tuple, time1))
    return hashes

# -------------------------------------------------------------------
# 2. MATCHING LOGIC (MOCK DATABASE FOR DEMONSTRATION)
# -------------------------------------------------------------------

# In a real scenario, you would pre-compute the library and store it in a dictionary.
# We will simulate this using Streamlit's session state to hold the "indexed" database.
if 'database' not in st.session_state:
    st.session_state.database = {} # Format: { hash_tuple: [ (song_name, time_offset), ... ] }

def match_audio(query_hashes):
    """Matches query hashes against the database and finds the alignment spike."""
    matches = {} # Format: { song_name: { offset: count } }
    
    for hash_tuple, query_time in query_hashes:
        if hash_tuple in st.session_state.database:
            for song_name, db_time in st.session_state.database[hash_tuple]:
                offset = db_time - query_time
                
                if song_name not in matches:
                    matches[song_name] = {}
                if offset not in matches[song_name]:
                    matches[song_name][offset] = 0
                
                matches[song_name][offset] += 1
                
    # Find the song with the highest alignment spike (most hashes at a single offset)
    best_song = None
    best_spike = 0
    all_spikes_for_best_song = {}
    
    for song, offsets in matches.items():
        max_spike_for_song = max(offsets.values()) if offsets else 0
        if max_spike_for_song > best_spike:
            best_spike = max_spike_for_song
            best_song = song
            all_spikes_for_best_song = offsets
            
    return best_song, best_spike, all_spikes_for_best_song

# -------------------------------------------------------------------
# 3. STREAMLIT USER INTERFACE
# -------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="EE200: Audio Fingerprinting")
st.title("EE200: Audio Fingerprinting")
st.markdown("Index a library of songs, then identify short clips against it.")

# Define the required tabs
tab_identify, tab_batch = st.tabs(["Identify a Clip", "Batch Mode"])

with tab_identify:
    st.header("Identify a single clip")
    uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3, FLAC)", type=['wav', 'mp3', 'flac'], key="single")
    
    if uploaded_file is not None:
        with st.spinner("Processing audio..."):
            # Load audio using librosa
            audio_data, fs = librosa.load(uploaded_file, sr=None, mono=True)
            
            # Step 1: Spectrogram
            f, t, Sxx_db = get_spectrogram(audio_data, fs)
            
            # Step 2: Constellation
            peaks = extract_peaks(Sxx_db)
            
            # Step 3: Hashes
            query_hashes = generate_hashes(peaks)
            
            # Note: Because the database is empty in this template, it won't find a match.
            # You need to implement the library indexing step.
            best_song, best_spike, offset_data = match_audio(query_hashes)
            
            # Display Results
            if best_song:
                st.success(f"**Match Found:** {best_song} (Confidence: {best_spike} matching hashes aligned)")
            else:
                st.warning("No match found in the current database. Have you indexed the library?")
                
            # Render Visualizations
            st.subheader("Intermediate Steps Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**1. Spectrogram**")
                fig, ax = plt.subplots(figsize=(8, 4))
                cax = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma')
                ax.set_ylabel('Frequency [Hz]')
                ax.set_xlabel('Time [sec]')
                fig.colorbar(cax, ax=ax, label='Intensity [dB]')
                st.pyplot(fig)
                
            with col2:
                st.markdown("**2. Constellation of Peaks**")
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                # Unzip peaks for plotting
                if peaks:
                    peak_freqs, peak_times = zip(*peaks)
                    # Convert indices back to actual time/frequency for plotting
                    ax2.scatter([t[i] for i in peak_times], [f[i] for i in peak_freqs], color='cyan', s=10)
                ax2.set_ylabel('Frequency [Hz]')
                ax2.set_xlabel('Time [sec]')
                ax2.set_facecolor('black')
                st.pyplot(fig2)

            st.markdown("**3. Alignment Spike (Histogram)**")
            if offset_data:
                fig3, ax3 = plt.subplots(figsize=(10, 3))
                ax3.bar(offset_data.keys(), offset_data.values(), width=2, color='orange')
                ax3.set_xlabel("Time Offset (Database frame - Query frame)")
                ax3.set_ylabel("Number of Aligning Hashes")
                st.pyplot(fig3)

with tab_batch:
    st.header("Identify multiple clips at once")
    batch_files = st.file_uploader("Upload query clips", type=['wav', 'mp3', 'flac'], accept_multiple_files=True, key="batch")
    
    if st.button("Run Batch"):
        if batch_files:
            results = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(batch_files):
                audio_data, fs = librosa.load(file, sr=None, mono=True)
                f, t, Sxx_db = get_spectrogram(audio_data, fs)
                peaks = extract_peaks(Sxx_db)
                query_hashes = generate_hashes(peaks)
                
                best_song, _, _ = match_audio(query_hashes)
                
                # Format required by the assignment
                filename = os.path.splitext(file.name)[0]
                prediction = best_song if best_song else "none"
                results.append({"filename": filename, "prediction": prediction})
                
                progress_bar.progress((i + 1) / len(batch_files))
                
            # Create DataFrame and display
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            # Export to CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download results.csv",
                data=csv,
                file_name='results.csv',
                mime='text/csv',
            )
        else:
            st.error("Please upload files to run the batch process.")