import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter
from collections import defaultdict
import os
import pickle

# -------------------------------------------------------------------
# 1. SETUP & DATABASE LOADING (Cloud-Safe)
# -------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="EE200: Audio Fingerprinting")

# Safely load the database using relative paths so it works on GitHub/Streamlit Cloud
if 'database' not in st.session_state:
    # Look for the file in the exact same folder as this script
    db_path = os.path.join(os.path.dirname(__file__), 'song_database.pkl')
    
    if os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            st.session_state.database = pickle.load(f)
    else:
        st.session_state.database = {}
        st.error(f"⚠️ Database not found at {db_path}. Please ensure 'song_database.pkl' is uploaded to your GitHub repository!")

# -------------------------------------------------------------------
# 2. SIGNAL PROCESSING & FINGERPRINTING LOGIC
# -------------------------------------------------------------------

def get_spectrogram(audio_data, fs, nperseg=1024):
    """Computes the spectrogram of the audio signal."""
    f, t, Sxx = signal.spectrogram(audio_data, fs, window='hann', nperseg=nperseg, noverlap=nperseg//2)
    Sxx_db = 10 * np.log10(Sxx + 1e-10) 
    return f, t, Sxx_db

def extract_peaks(Sxx_db, filter_size=20, threshold_db=10):
    """Finds local maxima (the constellation) in the spectrogram."""
    local_max = maximum_filter(Sxx_db, size=filter_size)
    peak_mask = (Sxx_db == local_max) & (Sxx_db > threshold_db)
    freq_idx, time_idx = np.where(peak_mask)
    return list(zip(freq_idx, time_idx))

def generate_hashes(peaks, fan_out=5):
    """Pairs nearby peaks to create robust hashes."""
    peaks = sorted(peaks, key=lambda x: x[1])
    hashes = []
    for i in range(len(peaks)):
        for j in range(1, fan_out + 1):
            if i + j < len(peaks):
                freq1, time1 = peaks[i]
                freq2, time2 = peaks[i + j]
                time_delta = time2 - time1
                hash_tuple = (freq1, freq2, time_delta)
                hashes.append((hash_tuple, time1))
    return hashes

# -------------------------------------------------------------------
# 3. OPTIMIZED MATCHING LOGIC
# -------------------------------------------------------------------

def match_audio(query_hashes):
    """Matches query hashes and computes the alignment histogram."""
    matches = defaultdict(lambda: defaultdict(int))
    
    for hash_tuple, query_time in query_hashes:
        if hash_tuple in st.session_state.database:
            for song_name, db_time in st.session_state.database[hash_tuple]:
                offset = db_time - query_time
                matches[song_name][offset] += 1
                
    best_song = None
    best_spike = 0
    all_spikes_for_best_song = {}
    
    for song_name, offsets in matches.items():
        if offsets:
            current_max = max(offsets.values())
            if current_max > best_spike:
                best_spike = current_max
                best_song = song_name
                all_spikes_for_best_song = offsets
            
    return best_song, best_spike, all_spikes_for_best_song

# -------------------------------------------------------------------
# 4. STREAMLIT USER INTERFACE
# -------------------------------------------------------------------

st.title("EE200: Audio Fingerprinting")
st.markdown("Index a library of songs, then identify short clips against it.")

tab_identify, tab_batch = st.tabs(["Identify a Clip", "Batch Mode"])

with tab_identify:
    st.header("Identify a single clip")
    uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3, FLAC)", type=['wav', 'mp3', 'flac'], key="single")
    
    if uploaded_file is not None:
        with st.spinner("Processing audio..."):
            audio_data, fs = librosa.load(uploaded_file, sr=None, mono=True)
            f, t, Sxx_db = get_spectrogram(audio_data, fs)
            peaks = extract_peaks(Sxx_db)
            query_hashes = generate_hashes(peaks)
            
            best_song, best_spike, offset_data = match_audio(query_hashes)
            
            if best_song:
                st.success(f"**Match Found:** {best_song} (Confidence: {best_spike} matching hashes aligned)")
            else:
                st.warning("No match found. The clip might not be in the database or the audio is too noisy.")
                
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
                if peaks:
                    peak_freqs, peak_times = zip(*peaks)
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
            else:
                st.info("No alignments to plot.")

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
                
                filename = os.path.splitext(file.name)[0]
                prediction = best_song if best_song else "none"
                results.append({"filename": filename, "prediction": prediction})
                
                progress_bar.progress((i + 1) / len(batch_files))
                
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download results.csv",
                data=csv,
                file_name='results.csv',
                mime='text/csv',
            )
        else:
            st.error("Please upload files to run the batch process.")