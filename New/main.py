# Imports
import sounddevice as sd
from scipy.io.wavfile import write
import pyaudio
import numpy as np
from openwakeword.model import Model
import argparse
import pyttsx3
import cv2



fs = 44100  # Sample rate
listenlength = 10  # Duration of recording




vid_cap = cv2.VideoCapture(0)
vid_fourcc = cv2.VideoWriter_fourcc(*'mp4v')

###################################
#FOR TEXT TO SPEECH################
###################################
engine = pyttsx3.init() # object creation

""" RATE"""
rate = engine.getProperty('rate')   # getting details of current speaking rate
                                    #printing current voice rate
engine.setProperty('rate', 125)     # setting up new voice rate


"""VOLUME"""
volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
                                        #printing current volume level
engine.setProperty('volume',1.0)        # setting up volume level  between 0 and 1

"""VOICE"""
voices = engine.getProperty('voices')       #getting details of current voice
#engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
engine.setProperty('voice', voices[0].id)   #changing index, changes voices. 1 for female


####################
#END TEXT TO SPEECH#
####################


# Parse input arguments
parser=argparse.ArgumentParser()
parser.add_argument(
    "--chunk_size",
    help="How much audio (in number of samples) to predict on at once",
    type=int,
    default=2560,
    required=False
)
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    default="hey jarvis",
    required=False
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite'",
    type=str,
    default='tflite',
    required=False
)

args=parser.parse_args()

# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Load pre-trained openwakeword models
if args.model_path != "":
    owwModel = Model(wakeword_models=[args.model_path], inference_framework=args.inference_framework)
else:
    owwModel = Model(inference_framework=args.inference_framework)

n_models = len(owwModel.models.keys())

# Run capture loop continuosly, checking for wakewords
if __name__ == "__main__":
    # Generate output string header
    print("\n\n")
    print("#"*100)
    print("Listening for wakewords...")
    print("#"*100)
    print("\n"*(n_models*3))
    disambiguate = 0
    while True:
        # Get audio
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Feed to openWakeWord model
        prediction = owwModel.predict(audio)

        # Column titles
        n_spaces = 16
        output_string_header = """
            Model Name         | Score | Wakeword Status
            --------------------------------------
            """

        for mdl in owwModel.prediction_buffer.keys():
            # Add scores in formatted table
            scores = list(owwModel.prediction_buffer[mdl])
            curr_score = format(scores[-1], '.20f').replace("-", "")

            output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {"--"+" "*20 if scores[-1] <= 0.3 else "Wakeword Detected!"}
            """
        if disambiguate > 0:
            disambiguate -= 1
        # Print results table
        if disambiguate == 0:
            if scores[-1] >= 0.6:
                disambiguate += 5
                print("Wakeword Detected!")
                print("Playing Sound!")

                vid_out = cv2.VideoWriter('./apiRequest/videofiles/vision.mp4', vid_fourcc, 20.0, (640,480))

                engine.say(f"I heard my name! I'm recording for {listenlength} seconds starting now!")
                engine.runAndWait()
                engine.stop()

                listenrecording = sd.rec(int(listenlength * fs), samplerate=fs, channels=2)   
                timerecorded = 0.0
                while(int(timerecorded)<listenlength):
                    # Capture each frame of webcam video
                    ret,frame = vid_cap.read()
                    timerecorded+=0.05 #20 fps
                    vid_out.write(frame)
                
                sd.wait()  # Wait until recording is finished
                vid_cap.release()#Close all the video stuff
                vid_out.release()#Close all the video stuff
                cv2.destroyAllWindows()#Close all the video stuff

                write('./apiRequest/audiofiles/listen.wav', fs, listenrecording)  # Save as WAV file

                engine.say('Done recording, let me think about that for a moment...')
                engine.runAndWait()
                engine.stop()

                engine.say('Returning to listening for my name...')
                engine.runAndWait()
                engine.stop()
                print("\n\n")
                print("#"*10)
                print("Listening for wakewords...")
                print("#"*10)
                print("\n\n")

                
                
                
        
