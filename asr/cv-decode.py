import sys, argparse
import requests
# import json
import pandas as pd
import os

def transcribe_audio(file_path):
    url = "http://localhost:8001/asr"
    try:
        # Open the audio file in binary mode
        with open(file_path, "rb") as audio_file:
            files = {"file": (file_path, audio_file, "audio/mpeg")}
            # Send a POST request with the file as multipart/form-data
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            # Parse and print the response
            data = response.json()
            # print(json.dumps(data, indent=4))
            
            transcription = data.get("transcription", None)
            duration = data.get("duration", None)
            
            return (transcription, duration)
        else:
            print(f"Server returned an error: {response.status_code}, {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to the server: {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")

    return None

def process_audio_csv(input_csv, audio_folder):

    if not os.path.isfile(input_csv):
        print(f"CSV file is not exist: {input_csv}")
  
    # Load the CSV file        
    df = pd.read_csv(input_csv)

    # Iterate through each row
    for index, row in df.iterrows():
        audio_file_path = f"{audio_folder}/{row['filename']}"

        if not os.path.isfile(audio_file_path):
            print(f"Audio file is not exist: {audio_file_path}")
            continue

        try:
            # Load the audio file
            result = transcribe_audio(audio_file_path)
            if result:
                df.at[index, 'duration'] = str(result[1])
                df.at[index, 'generated_text'] = result[0]

                # Delete the audio file after successful processing
                os.remove(audio_file_path)
        except Exception as e:
            print(f"Error processing file {audio_file_path}: {e}")
            # Set duration and transcription to None in case of error
            df.at[index, 'duration'] = None
            df.at[index, 'generated_text'] = None

    # Overwrite the original CSV file
    df.to_csv(input_csv, index=False)
    print(f"CSV file updated and saved to {input_csv}")


def parse_argv(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-a', '--audio_file_path', type=str, default=None, help='the input audio file for ASR')
    parser.add_argument(
        '-csv', '--input_csv', type=str, default='./cv-valid-dev/cv-valid-dev.csv', help='the input csv file to be processed')
    parser.add_argument(
    '-af', '--audio_folder', type=str, default=None, help='the input audio folder to be processed')    
    parser.add_argument(
        'argv', nargs=argparse.REMAINDER,
        help='Pass arbitrary arguments to the executable')
    
    args = parser.parse_args(argv)

    return args

if __name__ == "__main__":
    # Specify the path to your MP3 file
    args = parse_argv()
    if args.audio_file_path:
        result = transcribe_audio(args.audio_file_path)
        if result:
            print(f"Transcription: {result[0]}")
            print(f"Duration: {result[1]} seconds")
    elif args.audio_folder:
        process_audio_csv(args.input_csv, args.audio_folder)
    else:
        print('invalid input, run the program with parameter "--help" for help ')

