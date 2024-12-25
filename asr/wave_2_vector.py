#!/usr/bin/env python

import sys, argparse, os, logging, time
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import AutoTokenizer, AutoModel
import torch, librosa
import pathlib

import warnings
warnings.filterwarnings('ignore')

def get_log_dir(logdir='~/temp/log'):
    log_dir = os.path.expanduser(logdir)
    #log_dir = os.path.join(log_dir, time.strftime("%Y%m%d"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_dir

class asr_logger:
    def __init__( self, debug=True, logging_level = logging.INFO):
        self.debug = debug
        self.log_dir = get_log_dir()
        log_name = os.path.join(self.log_dir, 'asr.log')
        formater = '%(asctime)s %(message)s'
        try:
            logging.basicConfig(level= logging_level, #logging.DEBUG, logging.INFO or logging.WARNING
                                format=formater, filename=log_name, filemode='a')
        except:
            print('I/O error')
        
        rootLogger = logging.getLogger('')
        formatter = logging.Formatter(formater)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        rootLogger.addHandler(console)      
    def shutdown(self):
        logging.shutdown()
    def info(self, msg, *args, **kwargs):
        if self.debug:
            logging.info(msg, *args, **kwargs)
    def error(self, msg, *args, **kwargs):
        logging.error(msg, *args, **kwargs)
    def warn(self, msg, *args, **kwargs):
        logging.warning(msg, *args, **kwargs)            


model_list={
            '1': 'facebook/wav2vec2-large-960h',
            '2': 'wav2vec2-large-960h-cv', # fine-tuned model
            }

def parse_argv(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-a', '--audio_file', type=str, default='./asr/audio_sample/taken_clip.wav', help='the input audio file for ASR')
    parser.add_argument(
        '-b', dest='batch_recognition', action='store_true', help='batch_recognition for the input audio file')
    parser.add_argument(
        '-m', '--model_name', type=str, default='1', 
        help='the model name to be loaded, for download it needs specify full model name')
    parser.add_argument('--cpu', dest='use_cpu', action='store_true', default=False,
            help='use cpu for computing')
    parser.add_argument(
        "-dir", "--dataset_dir", type=str, default='~/temp/test_dataset', help="Path to location of dataset")    
    parser.add_argument(
        '-dst', dest='ds_testing', action='store_true', default=False, help='use test dataset to verify the model')    
    parser.add_argument(
        'argv', nargs=argparse.REMAINDER,
        help='Pass arbitrary arguments to the executable')
    
    args = parser.parse_args(argv)

    return args

class wav2vec2_asr:
    """
    ASR engine implementation with facebook/wav2vec2
    """
    def __init__(self, logger=None, audio_chunk_size=30):
        if logger is None:
            logger = asr_logger()
        self.log = logger
        self.audio_chunk_size = audio_chunk_size
    def batch_recognition(self, batchs):
        started_time = time.monotonic()

        inputs = self.processor(batchs, sampling_rate = 16000, return_tensors = 'pt', padding=True)
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # retrieve logits
        with torch.no_grad():
            #logits = self.model(input_values).logits
            logits = self.model(input_values, attention_mask=attention_mask).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions = self.processor.batch_decode(predicted_ids)

        #transcriptions = ' '.join(transcriptions)
        #transcriptions = transcriptions.lower()
        #print(transcriptions)
        
        delta_time = time.monotonic() - started_time
        #self.log.info('time consumed: {} sec'.format(delta_time))

        return transcriptions
    
    def load_modle(self, model_index='1', use_cpu=False):

        if model_index == '2': # fune-tuned mode 
            curr_file_dir = pathlib.Path(__file__).resolve().parent
            model_name = model_list.get(model_index, 'facebook/wav2vec2-large-robust-ft-libri-960h')
            model_name = os.path.join(curr_file_dir.parent, 'asr-train', model_name)
        else:
            model_name = model_list.get(model_index, 'facebook/wav2vec2-large-robust-ft-libri-960h')
        print('++++++ Selected model: {}'.format(model_name))        
        
        if use_cpu:
            self.device = 'cpu' # force to use cpu for computing
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # cuda:0 is first gpu device

        self.log.info('Use {} for computing'.format(self.device))
        # load model and processor
        self.log.info("Download the lage model will take for a while depend on your network speed if it's not cached.")
        self.log.info('Loading model WIP, please wait ...')

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.log.info('Model loaded !!')
    def chunk_audio_src(self, audio_file):
        audio_file = os.path.expanduser(audio_file)
        if os.path.isfile(audio_file):
            stream = librosa.stream(
                audio_file,
                block_length=self.audio_chunk_size,frame_length=16000,hop_length=16000
            )
            
            speechs=list()
            
            for speech in stream:
                speechs.append(speech)    
    
            return speechs            
        else:
            self.log.error('No audio file found:{}'.format(audio_file))

            return None

    def get_audio_src(self, audio_file=None):
        speech = None
        audio_file = os.path.expanduser(audio_file)
        if os.path.isfile(audio_file):
            speech, rate = librosa.load(audio_file, sr=16000)
        else:
            self.log.error('No audio source specified')

        return speech
    def single_steam_recognition(self, speech):
        #started_time = time.monotonic()
        recognized_transcript = ''
        #for speech in stream:
        if len(speech.shape) > 1:
            speech = speech[:, 0] + speech[:, 1]

        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        # attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            # logits = self.model(input_values, attention_mask=attention_mask).logits
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        # Passing the prediction to the processor decode to get the transcription
        transcription = self.processor.decode(predicted_ids[0])
        recognized_transcript += transcription
            
        #delta_time = time.monotonic() - started_time
        #self.log.info('consumed: {:.2f} sec, transcript:{}'.format(delta_time, recognized_transcript.lower()))

        return recognized_transcript.lower()
    def dataset_testing(self, dataset_dir):
        from datasets import load_from_disk
        import random
        
        path_exp = os.path.expanduser(dataset_dir)
        ds = load_from_disk(path_exp)
        print('ds.num_rows:{}'.format(ds.num_rows))
    
        while True:
            try:
                random_index = random.randrange(ds.num_rows-5)
                print('random_index:{}'.format(random_index))
                # Iterate all rows using DataFrame.iterrows()
                for index in range(4):
                    item = ds[index+random_index]
                    self.dataset_recognition(item)
                    print('-'*30)
                key = input('type any key---------')
            except KeyboardInterrupt:
                break            
        
    def dataset_recognition(self, ds_item):
        recognized_transcript = ''
        
        inputs = self.processor(ds_item["input_values"], sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        # attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        # Passing the prediction to the processor decode to get the transcription
        transcription = self.processor.decode(predicted_ids[0])
        recognized_transcript += transcription
        
        labeled_txt = self.processor.decode(ds_item["labels"])
        
        print('labeled_txt:{}'.format(labeled_txt.lower()))
        print('transcript:{}'.format(recognized_transcript.lower()))
        return recognized_transcript.lower()    
def main(argv=sys.argv[1:]):
    args = parse_argv()
    logger = asr_logger()
    engine = wav2vec2_asr(logger)

    
    engine.load_modle(args.model_name, args.use_cpu)

    if args.ds_testing:                             # dataset validation
        engine.dataset_testing(args.dataset_dir)
    elif args.batch_recognition:                            # perform ASR for large audio file
        batch = engine.chunk_audio_src(args.audio_file)
        if batch:
            trans= engine.batch_recognition(batch)
            transcript = ' '.join(trans)
            transcript = transcript.lower()
            logger.info('Recognized transcript is: {}'.format(transcript))
    else:                                            # perform ASR from mic input
        speech = engine.get_audio_src(args.audio_file)
        if speech is not None:
            transcript = engine.single_steam_recognition(speech)
            logger.info('transcript:{}'.format(transcript))


if __name__ == '__main__':
    main()