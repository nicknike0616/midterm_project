import numpy as np
import serial
import time

waitTime = 0.1
song_number =3
song_number =song_number/4
song_name=(["Little Star    ", "YAMAHA  ", "Little Bee     "])

song = np.array(
[
  262, 262, 392, 392, 440, 440, 392,

  349, 349, 330, 330, 294, 294, 262,

  392, 392, 349, 349, 330, 330, 294,

  392, 392, 349, 349, 330, 330, 294,

  262, 262, 392, 392, 440, 440, 392,

  349, 349, 330, 330, 294, 294, 262]
)

noteLength = np.array(
[
  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1,

  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1,

  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1,

  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1,

  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1,

  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1]
)

song1 = np.array(
[
  262, 294, 330, 349, 392, 440, 349, 330, 294, 262

  392, 349, 330, 392, 349, 330, 294,

  392, 349, 330, 392, 349, 330, 294,

  262, 294, 330, 349, 392, 440, 349, 330, 294, 262,

  0, 0, 0, 0, 0, 0, 0, 0]
)



noteLength1 = np.array(
[
  0.25, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.5, 1,

  1, 0.25, 0.25, 0.5, 0.5, 0.5, 1,

  1, 0.25, 0.25, 0.5, 0.5, 0.5, 1,

  0.25, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.5, 1,

  0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
)

song2 = np.array(
[
  392, 330, 330, 349, 294, 294,
  
  262, 294, 330, 349, 392, 392, 392,
  
  392, 330, 330, 349, 294, 294, 

  262, 330, 392, 392, 330,
  
  294, 294, 294, 330, 349,
  
  330, 330, 330, 349, 392,
  
  392, 330, 349, 294,
  
  262, 330, 392, 262]
)

noteLength2 =np.array(
[
  0.5, 0.5, 1, 0.5, 0.5, 1,
  
  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1,

  0.5, 0.5, 1, 0.5, 0.5, 1,

  0.5, 0.5, 0.5, 0.5, 2,

  1, 1, 0.5, 0.5, 1,
  
  1, 1, 0.5, 0.5, 1,

  0.5, 1.5, 0.5, 1.5,
  
  0.5, 0.5, 1, 2]
)

song = song /500
noteLength = noteLength /4
song1 = song1/500
noteLength1 = noteLength1/4
song2 = song2 /500
noteLength2 = noteLength2 /4

a = 1

formatter = lambda x: "%.3f" % x
formatter1 = lambda x: "%s" %x

serdev = '/dev/ttyACM0'

s = serial.Serial(serdev)

while a != 2:
    b = s.readline()
    print(b)
    c = (b)
    if b[0] == 49:
        print("Sending signal ...")
        for data in song:
            s.write(bytes(formatter(data), 'UTF-8'))
            time.sleep(waitTime)
        for data in noteLength:
            s.write(bytes(formatter(data), 'UTF-8'))
            time.sleep(waitTime)
        print("Signal sended")        

    if b[0] == 50:
        print("Sending signal ...")
        for data in song1:
            s.write(bytes(formatter(data), 'UTF-8'))
            time.sleep(waitTime)
        for data in noteLength1:
            s.write(bytes(formatter(data), 'UTF-8'))
            time.sleep(waitTime)
        print("Signal sended")

    if b[0] == 51:
        print("Sending signal ...")
        for data in song2:
            s.write(bytes(formatter(data), 'UTF-8'))
            time.sleep(waitTime)
        for data in noteLength2:
            s.write(bytes(formatter(data), 'UTF-8'))
            time.sleep(waitTime)
        #s.close()
        print("Signal sended")  

    if b[0] == 52:
        print("Sending the number of the songs ...")
        s.write(3)
        print("Signal sended")
        for data in song_name:
            s.write(bytes(formatter1(data), 'UTF-8'))
            print(formatter1(data))
            time.sleep(waitTime)
        print("Signal sended")