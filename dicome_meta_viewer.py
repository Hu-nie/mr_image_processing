import pydicom
import matplotlib.pyplot as plt


filename = './brain0126.dcm'

header = pydicom.dcmread(filename, stop_before_pixels=True)

print(header)


