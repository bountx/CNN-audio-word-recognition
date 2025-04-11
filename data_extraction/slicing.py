import os
import io
import asyncio
import logging
from glob import glob
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from google.cloud import speech

# --- Configuration ---
INPUT_FOLDER = '/home/andrzej/Documents/mcspeech/wavs'
OUTPUT_WAV_FOLDER = 'output_wav'      # Folder to store segmented .wav files
OUTPUT_MFCC_FOLDER = 'output_mfcc'    # Folder to store MFCC files (as .npy)
TSV_LABELS_FILE = 'labels.tsv'
MAX_FILES = 10000

SEGMENT_DURATION_SEC = 1.5          # Length of each segment in seconds
OVERLAP_DURATION_SEC = 0.75          # Overlap duration in seconds
SAMPLE_RATE = 16000                 # Uniform sample rate for all files
N_MFCC = 13                         # Number of MFCC features

LANGUAGE_CODE = "pl-PL"             # Polish language for speech recognition

# --- Setup Logging ---
logger = logging.getLogger("AsyncSpeechProcessing")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
# File handler
fh = logging.FileHandler("processing.log")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

# Create output directories if they don't exist
os.makedirs(OUTPUT_WAV_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_MFCC_FOLDER, exist_ok=True)

# Initialize Google Speech-to-Text client
speech_client = speech.SpeechClient()

# For holding all labels data from all files
all_labels_data = []

# Use a thread pool executor for blocking calls (I/O and CPU-heavy operations)
executor = ThreadPoolExecutor(max_workers=8)


async def transcribe_segment(segment_wav_path: str, sr: int) -> str:
    """
    Asynchronously transcribes the given WAV segment using Google Speech-to-Text.
    """
    loop = asyncio.get_running_loop()

    def sync_transcribe():
        with io.open(segment_wav_path, "rb") as audio_file:
            content = audio_file.read()
        audio_for_api = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sr,
            language_code=LANGUAGE_CODE
        )
        try:
            response = speech_client.recognize(config=config, audio=audio_for_api)
            transcription = " ".join(
                [result.alternatives[0].transcript for result in response.results]
            )
        except Exception as e:
            logger.error(f"Error transcribing {segment_wav_path}: {e}")
            transcription = ""
        return transcription

    transcription = await loop.run_in_executor(executor, sync_transcribe)
    return transcription


async def process_segment(segment, sr, base_name, segment_index, file_full_path):
    """
    Process a single segment: save as WAV, compute MFCCs, call transcription asynchronously.
    Returns tuple: (segment filename, start_time, end_time, transcription)
    """
    loop = asyncio.get_running_loop()

    segment_wav_name = f"{base_name}_seg{segment_index}.wav"
    segment_wav_path = os.path.join(OUTPUT_WAV_FOLDER, segment_wav_name)
    mfcc_filename = f"{base_name}_seg{segment_index}.npy"
    mfcc_path = os.path.join(OUTPUT_MFCC_FOLDER, mfcc_filename)

    # Save segment as WAV using soundfile in executor
    await loop.run_in_executor(executor, sf.write, segment_wav_path, segment, sr)

    def compute_mfcc():
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        # Concatenate along feature axis
        features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)
        np.save(mfcc_path, features)

    await loop.run_in_executor(executor, compute_mfcc)

    # Determine start and end times based on segment index and parameters.
    step_length = int((SEGMENT_DURATION_SEC - OVERLAP_DURATION_SEC) * sr)
    start_sample = segment_index * step_length
    end_sample = start_sample + int(SEGMENT_DURATION_SEC * sr)
    start_time = start_sample / sr
    end_time = end_sample / sr

    # Transcribe this segment asynchronously
    transcription = await transcribe_segment(segment_wav_path, sr)
    logger.info(f"Processed segment {segment_wav_name} from {file_full_path}: '{transcription}'")

    return [segment_wav_name, start_time, end_time, transcription]

async def process_file(wav_file: str):
    loop = asyncio.get_running_loop()
    base_name = os.path.splitext(os.path.basename(wav_file))[0]
    logger.info(f"Processing file: {wav_file}")

    # Load and resample the audio
    audio, sr = await loop.run_in_executor(executor, lambda: librosa.load(wav_file, sr=SAMPLE_RATE))
    total_samples = len(audio)
    segment_length = int(SEGMENT_DURATION_SEC * sr)
    step_length = int((SEGMENT_DURATION_SEC - OVERLAP_DURATION_SEC) * sr)

    segment_tasks = []
    segment_index = 0

    # Ensure we always process at least one segment
    if total_samples < segment_length:
        # Pad full audio with zeros (silence) to reach segment length
        padded_segment = np.pad(audio, (0, segment_length - total_samples))
        task = asyncio.create_task(process_segment(padded_segment, sr, base_name, segment_index, wav_file))
        segment_tasks.append(task)
    else:
        for start_sample in range(0, total_samples, step_length):
            end_sample = start_sample + segment_length
            segment = audio[start_sample:end_sample]

            # Pad the segment if it's shorter than the desired length
            if len(segment) < segment_length:
                segment = np.pad(segment, (0, segment_length - len(segment)))

            task = asyncio.create_task(process_segment(segment, sr, base_name, segment_index, wav_file))
            segment_tasks.append(task)
            segment_index += 1

    # Await all segment tasks for this file
    segments_data = await asyncio.gather(*segment_tasks)
    logger.info(f"Completed file: {wav_file} with {len(segments_data)} segments.")
    return segments_data

async def main():
    # Retrieve a list of WAV files (only first MAX_FILES)
    wav_files = glob(os.path.join(INPUT_FOLDER, '*.wav'))[:MAX_FILES]
    logger.info(f"Found {len(wav_files)} files to process.")

    file_tasks = [asyncio.create_task(process_file(wav_file)) for wav_file in wav_files]

    # Gather results from files; results is a list of lists
    results = await asyncio.gather(*file_tasks)
    
    # Flatten results (each file returns a list of segment data)
    for file_result in results:
        all_labels_data.extend(file_result)

    # Save the final TSV file with labels
    labels_df = pd.DataFrame(all_labels_data, columns=["segment_filename", "start_time", "end_time", "transcription"])
    labels_df.to_csv(TSV_LABELS_FILE, sep='\t', index=False)
    logger.info(f"Saved labels to {TSV_LABELS_FILE}")
    logger.info("Processing complete.")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception(f"Unhandled exception in processing: {e}")
