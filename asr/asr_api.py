# import threading
# import os
import io
import time
import traceback
from flask import Flask, request, jsonify
# import sounddevice as sd
import librosa
from wave_2_vector import wav2vec2_asr

PORT = 8001
HOST = '0.0.0.0'

class ASRServer:
    def __init__(self):
        self.asr_model_init()
        self.app = Flask(__name__)

    def asr_model_init(self):
        """Initialize the ASR model."""
        self.asr = wav2vec2_asr()
        self.asr.load_modle()

    def asr_transcribe(self, audio_data, play=False, save_wav=False):
        """Transcribe the audio data and optionally play or save the audio."""
        wav_buf = io.BytesIO()
        wav_buf.name = 'temp_audio'
        wav_buf.write(audio_data)
        wav_buf.seek(0)

        # Load audio with librosa
        wav, sr = librosa.load(wav_buf, sr=16000)

        if wav is not None:
            # Perform transcription
            transcript = self.asr.single_steam_recognition(wav)
            print(f'Transcription: {transcript}')

            # Calculate duration
            duration = len(wav) / sr
            print(f'Duration: {duration:.1f} seconds')

            # # Playback
            # if play:
            #     print('Playing sound...')
            #     sd.play(wav, sr)
            #     sd.wait()

            # Save WAV file
            if save_wav:
                wav_buf.seek(0)
                filename = f"{str(time.time()).split('.')[0]}.wav"
                with open(filename, "wb") as outfile:
                    outfile.write(wav_buf.getbuffer())

            # Return transcription and duration
            return {
                "transcription": transcript,
                "duration": f"{duration:.1f}"
            }
        else:
            raise ValueError("Invalid audio format!")

    def run(self):
        """Run the Flask server."""
        @self.app.route('/asr', methods=['POST'])
        def asr():
            try:
                # Extract file from the multipart form-data request
                if 'file' not in request.files:
                    return jsonify({"error": "No file part in the request"}), 400
                file = request.files['file']
                if file.filename == '':
                    return jsonify({"error": "No file selected"}), 400

                # Read binary data from the file
                audio_data = file.read()
                
                # Process and transcribe the audio
                result = self.asr_transcribe(audio_data)
                return jsonify(result), 200

            except Exception as e:
                traceback.print_exc()
                return jsonify({"error": str(e)}), 500

        # Define a route to retrieve data
        @self.app.route('/ping', methods=['GET'])
        def ping():
            return jsonify({"message": "pong"}), 500

        self.app.run(host=HOST, port=PORT, debug=True, use_reloader=False)

def main():
    server = ASRServer()
    server.run()

if __name__ == '__main__':
    main()
