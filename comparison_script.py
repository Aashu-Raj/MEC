import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# ——— load your two .mat files ———
base = sio.loadmat('result_base_dynamic_users.mat')
trans = sio.loadmat('result_transformer_dynamic_users.mat')

# ——— pull out (T×N) arrays, then average across users (axis=1) ———
qb = base['data_queue'].mean(axis=1)
qt = trans['data_queue'].mean(axis=1)

eb = base['energy_queue'].mean(axis=1)
et = trans['energy_queue'].mean(axis=1)

# ——— pull out training‐loss vectors (must exist in your .mat) ———
# if your key is different, e.g. 'loss_history' or 'train_loss_his', change it here:
lb = base['data_arrival'].flatten()     
lt = trans['data_arrival'].flatten()

# ——— common time axis ———
T = len(qb)
t = np.arange(T)

# ——— Fig 1: average data‐queue ———
plt.figure(figsize=(15,8))
plt.plot(t, qb,  linewidth=1.0, label='Base')
plt.plot(t, qt,  linewidth=1.0, label='Transformer')

plt.xlabel('Time Frames')
plt.ylabel('Avg data‑queue')
plt.title('Avg Data‑Queue: Base vs Transformer')
plt.legend()
plt.tight_layout()

# ——— Fig 2: average energy‑queue ———
plt.figure(figsize=(15,8))
plt.plot(t, eb, linewidth=1.0, label='Base')
plt.plot(t, et, linewidth=1.0, label='Transformer')
plt.xlabel('Time Frames')
plt.ylabel('Avg energy‑queue')
plt.title('Avg Energy‑Queue: Base vs Transformer')
plt.legend()
plt.tight_layout()

# ——— Fig 3: training losses ———
plt.figure(figsize=(15,8))
plt.plot(lb,  linewidth=0.8, label='Base')
plt.plot(lt,  linewidth=0.8, label='Transformer')
plt.xlabel('Training epoch')
plt.ylabel('Training loss')
plt.title('Training Loss: Base vs Transformer')
plt.legend()
plt.tight_layout()

plt.show()
