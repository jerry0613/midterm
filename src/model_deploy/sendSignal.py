import numpy as np

import serial

import time


waitTime = 0.1


# generate the waveform table

signalLength = 42

t = np.linspace(0, 2*np.pi, signalLength)

signalTable = [
 [261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261],
  [],
  [],
  [],
  []]

noteTable = [
 [1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2],
  [],
  [],
  [],
  []]


# output formatter

formatter = lambda x: "%d" % x

serdev = '/dev/ttyACM0'
s = serial.Serial(serdev)
line=s.readline()
cur = int(line)

print("Sending Song%d" % (cur+1))

print("It may take about %d seconds ..." % (int(signalLength * waitTime * 2)))

for data in signalTable[cur]:

  s.write(bytes(formatter(data), 'UTF-8'))

  time.sleep(waitTime)

for data in noteTable[cur]:

  s.write(bytes(formatter(data), 'UTF-8'))

  time.sleep(waitTime)

s.close()

print("Signal sended")