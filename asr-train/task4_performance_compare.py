import sys, argparse
import pandas as pd

import librosa

import os
import gc

# Add the root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
# Now import wave_2_vector
from asr.wave_2_vector import *
from asr.wave_2_vector import wav2vec2_asr

from evaluate_fine_tuning import wer_evaluation

def parse_argv(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '-csv', '--csv_path', type=str, default='../../common_voice/cv-valid-dev.csv', 
        help=" Path to the input CSV file")    
    parser.add_argument(
    '-audio', '--audio_path', type=str, default='../../common_voice/cv-valid-dev', 
    help=" Path to the directory containing audio files")
    parser.add_argument(
        'argv', nargs=argparse.REMAINDER,
        help='Pass arbitrary arguments to the executable')
    
    args = parser.parse_args(argv)

    return args

class transcribe_cv:
    def __init__(self, model_index='1'):
        self.asr_model_init(model_index)
    def asr_model_init(self, model_index):
        """Initialize the ASR model."""
        self.asr = wav2vec2_asr()
        self.asr.load_modle(model_index)
    def single_file_transcription(self, file_path):
        # Load audio with librosa
        wav, sr = librosa.load(file_path, sr=16000)

        if wav is not None:
            # Perform transcription
            transcription = self.asr.single_steam_recognition(wav)
            # print(f'Transcription: {transcription}')
            return transcription

        return None
    def cv_transcription(self,
                        csv_path="../../common_voice/cv-valid-dev.csv", 
                        audio_path="../../common_voice/cv-valid-dev",
                        new_column_name="generated_text", 
                        new_csv_file="cv-valid-dev_ft.csv"):

        # Function to update a CSV file with audio transcription results.

        # Args:
        #     csv_path (str): Path to the input CSV file.
        #     audio_path (str): Path to the directory containing audio files.
        #     new_column_name (str): Name of the new column to store transcriptions.
        #     new_csv_file (str): Path to save the updated CSV file.


        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_path)
        
        # Create a new column for storing transcriptions
        df[new_column_name] = None

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            filename = row['filename']  # Get the filename from the row
            audio_file = f"{audio_path}/{filename}"  # Construct the full path to the audio file

            transcription = self.single_file_transcription(audio_file)
            if transcription:        
                # Update the new column with the transcription
                df.at[index, new_column_name] = transcription

        # Save the updated DataFrame to a new CSV file
        df.to_csv(new_csv_file, index=False)
        print("Transciption add to column {}, save at {}".format(new_column_name, new_csv_file))


def main():
    args = parse_argv()

    if not os.path.isfile("cv-valid-dev_ft.csv"):
        print("Transcription using orginal wav2vec2-large-960h")

        tc = transcribe_cv('1') # 1 is for using 'facebook/wav2vec2-large-960h'

        tc.cv_transcription(args.csv_path, args.audio_path, "generated_text", "./cv-valid-dev_ft.csv")

        del tc
        gc.collect()

        print("Transcription using fine-tuned model")

        tc = transcribe_cv('2') # 2 is for loading  fune-tuned mode

        tc.cv_transcription("cv-valid-dev_ft.csv", args.audio_path, "generated_text_ft", "cv-valid-dev_ft.csv")

    # performance eval for the transcription which created by orinal model
    wer_evaluation("cv-valid-dev_ft.csv", transcription_col="generated_text")    

    # performance eval for the transcription which created by fine tuned model
    wer_evaluation("cv-valid-dev_ft.csv", transcription_col="generated_text_ft")

    # if not os.path.isfile("cv-valid-test/cv-valid-test.csv"):
    #     print("Transcription using fine-tuned model")

    #     tc = transcribe_cv('2') # 2 is for loading  fune-tuned mode

    #     tc.cv_transcription(args.csv_path, args.audio_path, "generated_text", "cv-valid-test/cv-valid-test.csv")

    #     # performance eval for the transcription which created by fine tuned model
    #     wer_evaluation("cv-valid-test/cv-valid-test.csv", transcription_col="generated_text")   

if __name__ == '__main__':
    main()