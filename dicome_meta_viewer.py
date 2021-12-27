import pydicom
import matplotlib.pyplot as plt


filename = './50_20 tof/10001.dcm'

header = pydicom.dcmread(filename, stop_before_pixels=True)

print(header)


