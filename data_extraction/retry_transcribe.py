import os
import io
import logging
import time
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
# import soundfile as sf # Not strictly needed for transcription retry
from google.cloud import speech
from google.api_core import exceptions as google_exceptions

# --- Configuration ---
LOG_FILE_PATH = 'processing.log'
TSV_LABELS_FILE = 'labels.tsv'
OUTPUT_WAV_FOLDER = 'output_wav'
SAMPLE_RATE = 16000
LANGUAGE_CODE = "pl-PL"

# Delay between *different file* API calls
API_CALL_DELAY = 0.5  # Keep a small delay between *successful* calls or moves to next file

# Retry Configuration
MAX_RETRIES = 3       # Number of retries on errors (including quota after long wait)
RETRY_DELAY = 5       # Seconds to wait before retrying *non-quota* errors
QUOTA_RETRY_DELAY = 65 # ****** Seconds to wait specifically after a 429 error ******

# --- Setup Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(log_formatter)
# File handler for this specific script's run
fh_retry = logging.FileHandler("retry_transcription_with_wait.log") # New log file name
fh_retry.setLevel(logging.INFO)
fh_retry.setFormatter(log_formatter)

retry_logger = logging.getLogger("RetryTranscriptionWithWait") # New logger name
retry_logger.setLevel(logging.INFO)
if not retry_logger.handlers:
    retry_logger.addHandler(ch)
    retry_logger.addHandler(fh_retry)

# --- Google Cloud Client ---
try:
    speech_client = speech.SpeechClient()
except Exception as e:
    retry_logger.critical(f"Failed to initialize Google Speech Client: {e}")
    retry_logger.critical("Ensure your GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly.")
    exit(1)

# Use a thread pool executor - keep workers low if quota is an issue
executor = ThreadPoolExecutor(max_workers=2) # Reduced workers


def find_failed_files(log_path):
    """Parses the log file to find filenames hit by 429 errors."""
    failed_files = set()
    regex = r"Error transcribing.*?([\w\-\.]+\.wav).*?: 429 Quota exceeded"
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(regex, line)
                if match:
                    filename = os.path.basename(match.group(1).strip())
                    failed_files.add(filename)
    except FileNotFoundError:
        retry_logger.error(f"Log file not found: {log_path}")
    except Exception as e:
        retry_logger.error(f"Error reading log file {log_path}: {e}")

    retry_logger.info(f"Found {len(failed_files)} unique files with 429 errors in {log_path}.")
    return list(failed_files)


def transcribe_single_wav(segment_wav_path: str, sr: int) -> str:
    """
    Transcribes a single WAV file with retries and specific long delays for quota errors.
    Returns transcription string or None if fails permanently.
    """
    filename = os.path.basename(segment_wav_path)
    if not os.path.exists(segment_wav_path):
        retry_logger.warning(f"WAV file not found: {segment_wav_path}")
        return None

    for attempt in range(MAX_RETRIES + 1):
        try:
            retry_logger.debug(f"Attempt {attempt + 1}/{MAX_RETRIES + 1} for {filename}")
            with io.open(segment_wav_path, "rb") as audio_file:
                content = audio_file.read()

            audio_for_api = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sr,
                language_code=LANGUAGE_CODE
            )

            future = executor.submit(speech_client.recognize, config=config, audio=audio_for_api)
            response = future.result()

            transcription = " ".join(
                [result.alternatives[0].transcript for result in response.results if result.alternatives]
            )
            retry_logger.debug(f"Successfully transcribed {filename} on attempt {attempt + 1}: '{transcription}'")
            return transcription # Success! Exit the loop and function.

        except google_exceptions.ResourceExhausted as e:
            # ****** Specifically handle quota errors with a LONG wait ******
            retry_logger.warning(f"Quota error (429) on attempt {attempt + 1} for {filename}: {e}")
            if attempt < MAX_RETRIES:
                retry_logger.warning(f"Waiting {QUOTA_RETRY_DELAY} seconds before retrying {filename} due to quota limit...")
                time.sleep(QUOTA_RETRY_DELAY)
                # Continue to the next iteration of the loop
            else:
                retry_logger.error(f"Quota error persisted after {MAX_RETRIES + 1} attempts for {filename}. Giving up on this file.")
                return None # Failed after retries

        except Exception as e:
            # Handle other errors with a shorter wait
            retry_logger.warning(f"Non-quota error on attempt {attempt + 1} for {filename}: {e}")
            if attempt < MAX_RETRIES:
                wait_time = RETRY_DELAY * (attempt + 1) # Optional: exponential backoff
                retry_logger.warning(f"Waiting {wait_time} seconds before retrying {filename}...")
                time.sleep(wait_time)
                # Continue to the next iteration of the loop
            else:
                retry_logger.error(f"Non-quota error persisted after {MAX_RETRIES + 1} attempts for {filename}. Giving up on this file.")
                return None # Failed after retries

    # This line should theoretically not be reached if logic is correct, but serves as a final fallback
    retry_logger.error(f"Exited retry loop unexpectedly for {filename}. Giving up.")
    return None


def main():
    failed_files = find_failed_files(LOG_FILE_PATH)
    if not failed_files:
        retry_logger.info("No files with 429 errors found in the log. Exiting.")
        return

    try:
        labels_df = pd.read_csv(TSV_LABELS_FILE, sep='\t')
        if 'segment_filename' not in labels_df.columns:
             retry_logger.error(f"Column 'segment_filename' not found in {TSV_LABELS_FILE}. Please check the file format.")
             return
    except FileNotFoundError:
        retry_logger.error(f"Labels file not found: {TSV_LABELS_FILE}")
        return
    except Exception as e:
        retry_logger.error(f"Error reading labels TSV {TSV_LABELS_FILE}: {e}")
        return

    updated_count = 0
    failed_update_count = 0
    files_processed_count = 0

    retry_logger.info(f"Attempting to re-transcribe {len(failed_files)} files with retry logic...")

    total_files = len(failed_files)
    for filename in failed_files:
        files_processed_count += 1
        retry_logger.info(f"--- Processing file {files_processed_count}/{total_files}: {filename} ---")
        segment_path = os.path.join(OUTPUT_WAV_FOLDER, filename)

        target_rows = labels_df[labels_df['segment_filename'] == filename]

        if target_rows.empty:
            retry_logger.warning(f"Filename {filename} from log not found in {TSV_LABELS_FILE}. Skipping.")
            continue
        # Handling multiple entries warning is still valid
        elif len(target_rows) > 1:
             retry_logger.warning(f"Multiple entries found for {filename} in {TSV_LABELS_FILE}. Will attempt to update all.")

        # Call the updated transcription function
        new_transcription = transcribe_single_wav(segment_path, SAMPLE_RATE)

        if new_transcription is not None:
            row_indices = target_rows.index
            original_transcriptions = labels_df.loc[row_indices, 'transcription'].tolist()
            labels_df.loc[row_indices, 'transcription'] = new_transcription
            updated_count += len(row_indices)
            retry_logger.info(f"Successfully updated {filename}. Old: '{original_transcriptions}', New: '{new_transcription}'")
        else:
            # Failure message now comes from transcribe_single_wav
            failed_update_count += len(target_rows)

        # Optional: Keep a small delay between processing *different* files
        # This helps even if the internal retry handles quota, prevents hammering
        # the API immediately after a successful transcription.
        if new_transcription is not None: # Only delay slightly after success/moving on
             retry_logger.debug(f"Waiting {API_CALL_DELAY} seconds before next file...")
             time.sleep(API_CALL_DELAY)

    retry_logger.info("--- Re-transcription Phase Complete ---")
    retry_logger.info(f"Successfully updated transcriptions for {updated_count} entries.")
    retry_logger.info(f"Failed to update transcriptions for {failed_update_count} entries after retries.")

    if updated_count > 0:
        try:
            labels_df.to_csv(TSV_LABELS_FILE, sep='\t', index=False)
            retry_logger.info(f"Successfully saved updated labels to {TSV_LABELS_FILE}")
        except Exception as e:
            retry_logger.error(f"CRITICAL: Failed to save updated labels to {TSV_LABELS_FILE}: {e}")
            retry_logger.error("The updated data might be lost. Check permissions and disk space.")
    else:
        retry_logger.info("No transcriptions were successfully updated, labels file remains unchanged.")


if __name__ == '__main__':
    main()