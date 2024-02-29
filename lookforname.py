import string
import time


while True:
    heardname = False
    f = open("./Transcriptions/transcription.txt", "r")
    lines = f.readlines()

    for line in lines:
        words = line.split()
        for word in words:
            if word.lower().translate(str.maketrans('', '', string.punctuation)) == ('r2d2' or 'r2c2' or 'r2b2' or '2d2'):
                heardname = True
    
    if heardname:
        print("I heard my name!!!!")
    else:
        print("I didn't hear my name!!!")
    #Edit sleep duration to adjust the rate at which it reads from the file
    time.sleep(1)
