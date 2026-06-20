import os
import pickle
import librosa
from app import get_spectrogram, extract_peaks, generate_hashes

def index_library(library_path):
    database = {}
    print("Starting indexing... this may take a moment.")
    
    for filename in os.listdir(library_path):
        if filename.endswith(('.wav', '.mp3', '.flac')):
            song_name = os.path.splitext(filename)[0]
            file_path = os.path.join(library_path, filename)
            
            # Extract hashes
            audio, fs = librosa.load(file_path, sr=None, mono=True)
            f, t, Sxx_db = get_spectrogram(audio, fs)
            peaks = extract_peaks(Sxx_db)
            hashes = generate_hashes(peaks)
            
            # Map hashes to song name and time
            for hash_tuple, time_offset in hashes:
                if hash_tuple not in database:
                    database[hash_tuple] = []
                database[hash_tuple].append((song_name, time_offset))
            print(f"Indexed: {song_name}")
            
    with open('song_database.pkl', 'wb') as f:
        pickle.dump(database, f)
    print("Database saved as song_database.pkl!")

# Execute the indexing
if __name__ == "__main__":
    index_library(r'C:\Users\Kavya\Documents\ee200\Q3\songs')