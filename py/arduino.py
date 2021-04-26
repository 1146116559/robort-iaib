import serial 
import time
import sys


arduino = serial.Serial('/dev/ttyUSB1', 115200, timeout=1)


while True:
    while arduino.in_waiting:
        try:
            data_raw = arduino.readline()
            data = data_raw.decode()
            if data:
                print(data)
                arduino.write(b'Hi Arduino\n')
        except:
            arduino.close() 