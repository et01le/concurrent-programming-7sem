import numpy as np
import matplotlib.pyplot as plt

sample_size = 4096
boundary = 100
arg_values = np.linspace(0, 1, boundary)
spectrum = np.fromfile(file = "spectrum.txt", dtype = float, count = sample_size, sep = ' ')[:boundary]

plt.plot(arg_values, spectrum)
plt.suptitle("sinc")
plt.savefig("spectrum.png")
