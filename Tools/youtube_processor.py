from __future__ import unicode_literals
import yt_dlp
import librosa
import soundfile as sf
import os
import sys
import subprocess
import logging
import time
import shutil
import pandas as pd
from tqdm import tqdm

# ---------- Setup ----------
os.makedirs("downloads", exist_ok=True)
os.makedirs("trimmed", exist_ok=True)
os.makedirs("separated", exist_ok=True)

logging.basicConfig(
    filename='process.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# ---------- Utilities ----------
def clean_filename(name):
    return "_".join(str(name).strip().split()).replace("/", "_")

def convert_time_to_seconds(t):
    try:
        if pd.isna(t): return 0
        if isinstance(t, (int, float)): return float(t)
        parts = str(t).strip().split(":")
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(float, parts)
            return m * 60 + s
        else:
            return float(parts[0])
    except:
        return 0

def get_column(row, options):
    for col in options:
        if col in row and not pd.isna(row[col]):
            return row[col]
    return None

# ---------- Step 1: Download from YouTube ----------
def download_from_url(url, name):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'downloads/{name}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'noplaylist': True,
        'quiet': False,
        'ignoreerrors': True,
        'postprocessor_args': ['-ar', '44100'],
        'preferredquality': '192',
    }

    logging.info("[1/3] Downloading from YouTube...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            output_file = os.path.join("downloads", f"{name}.wav")

            if not os.path.exists(output_file) or os.path.getsize(output_file) < 1024:
                logging.error("Downloaded file is empty or too small.")
                return None

        with tqdm(total=100, desc="Downloading", unit="%", ncols=80) as pbar:
            for _ in range(30):
                time.sleep(0.03)
                pbar.update(2)
        logging.info(f"✅ Download complete: {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"[ERROR] Download failed: {e}")
        return None

# ---------- Step 2: Trim Audio ----------
def trim_audio(input_path, start_sec, end_sec, name):
    try:
        logging.info("[2/3] Trimming audio...")
        with tqdm(total=100, desc="Trimming", ncols=80) as pbar:
            y, sr = librosa.load(input_path, sr=None)
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            trimmed = y[start_sample:end_sample]
            output_path = os.path.join("trimmed", f"{name}_trimmed.wav")
            sf.write(output_path, trimmed, sr)
            for _ in range(25):
                time.sleep(0.01)
                pbar.update(4)
        logging.info(f"✅ Trimmed audio saved: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"[ERROR] Trimming failed: {e}")
        return None

# ---------- Step 3: Run Demucs ----------
def separate_audio(audio_path, name):
    try:
        logging.info("[3/3] Running Demucs...")
        subprocess.run(
           [sys.executable, "-m", "demucs", audio_path, "--out", "separated", "--two-stems", "vocals"],
           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        model_dir = next(d for d in os.listdir("separated") if os.path.isdir(os.path.join("separated", d)) and d not in ["vocals", "no_vocals"])
        base = os.path.splitext(os.path.basename(audio_path))[0]
        out_dir = os.path.join("separated", model_dir, base)

        os.makedirs("separated/vocals", exist_ok=True)
        os.makedirs("separated/no_vocals", exist_ok=True)

        os.replace(os.path.join(out_dir, "vocals.wav"), f"separated/vocals/{name}_vocals.wav")
        os.replace(os.path.join(out_dir, "no_vocals.wav"), f"separated/no_vocals/{name}_no_vocals.wav")

        shutil.rmtree(os.path.join("separated", model_dir))

        logging.info(f"✅ Saved: separated/vocals/{name}_vocals.wav")
        logging.info(f"✅ Saved: separated/no_vocals/{name}_no_vocals.wav")

    except Exception as e:
        logging.error(f"[ERROR] Demucs failed: {e}")

# ---------- Manual Sheet Selection ----------
def process_single_sheet(file_path, sheet_name):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        logging.info(f"📄 Loaded sheet: {sheet_name} — {len(df)} rows")

        for index, row in df.iterrows():
            try:
                url = get_column(row, ['YouTube Link', 'URL', 'YT Link'])
                if not url or str(url).strip() == '':
                    continue

                processed_flag = get_column(row, ['Processed till source separation (y/n)', 'Processed', 'Done'])
                if str(processed_flag).lower() == 'y':
                    continue

                name = clean_filename(
                    get_column(row, ['Name', 'Title', 'Track Name']) or
                    f"{get_column(row, ['Raga'])}_{get_column(row, ['Artist name (initials)', 'Artist'])}" or
                    f"{sheet_name}_track_{index+1}"
                )

                start_time = get_column(row, ['Start Time (s)', 'Start Time', 'Start'])
                end_time = get_column(row, ['End Time (s)', 'End Time', 'End'])

                start_sec = convert_time_to_seconds(start_time)
                end_sec = convert_time_to_seconds(end_time)
                if end_sec <= start_sec:
                    end_sec = start_sec + 60

                logging.info(f"\n[{sheet_name}] Track {index+1}: {name}")
                logging.info(f"URL: {url}")
                logging.info(f"Time: {start_sec}s to {end_sec}s")

                wav_path = download_from_url(url, name)
                if not wav_path:
                    continue

                trimmed_path = trim_audio(wav_path, start_sec, end_sec, name)
                if not trimmed_path:
                    continue

                separate_audio(trimmed_path, name)

                df.at[index, 'Processed'] = 'y'

            except Exception as e:
                logging.error(f"❌ Failed row {index+1}: {e}")
                continue

        df.to_excel(f"{sheet_name}_processed.xlsx", index=False)
        logging.info(f"✅ Saved updated sheet: {sheet_name}_processed.xlsx")

    except Exception as e:
        logging.error(f"❌ Failed to process sheet '{sheet_name}': {e}")

# ---------- Main ----------
def main():
    excel_file = "songs 01.xlsx"
    sheet_name = input("🔢 Enter sheet name to process: ").strip()
    if not os.path.exists(excel_file):
        logging.error(f"Excel file not found: {excel_file}")
        return
    process_single_sheet(excel_file, sheet_name)

if __name__ == "__main__":
    main()
