import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

time_slots = [i for i in range(10000)]

# Load your .mat file
mat = scipy.io.loadmat('result_transformer_15.mat')
data_queue = mat['data_queue']
rate = mat['rate']
energy = mat['energy_consumption']

df_qlen = pd.DataFrame(data_queue)
df_rate = pd.DataFrame(rate)
df_energy = pd.DataFrame(energy)

plt.figure()

plt.title("Energy consumption")

plt.plot(time_slots,df_qlen.mean(axis=1),label="transformer")
plt.xlabel('Time')
plt.ylabel('energy')

plt.legend()

plt.savefig("energy consumption.png")

print(mat.keys())
