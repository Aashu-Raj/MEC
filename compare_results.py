# import scipy.io

# # Load .mat files
# base_data = scipy.io.loadmat('result_base_dynamic_users.mat')
# transformer_data = scipy.io.loadmat('result_transformer_dynamic_users.mat')

# # Calculate total data‐queue backlog
# total_queue_base        = base_data['data_queue'].sum()
# total_queue_transformer = transformer_data['data_queue'].sum()


# # Calculate total energy
# total_energy_base = base_data['energy_queue'].sum()
# total_energy_transformer = transformer_data['energy_queue'].sum()

# # Print the results
# print(f"Total Data‐Queue (Base):        {total_queue_base}")
# print(f"Total Data‐Queue (Transformer): {total_queue_transformer}")

# # Print the results
# print(f"Total Energy Consumption (Base): {total_energy_base}")
# print(f"Total Energy Consumption (Transformer): {total_energy_transformer}")
# import scipy.io
# import numpy as np

# # your two files
# base_path  = 'result_base_dynamic_users.mat'
# trans_path = 'result_transformer_dynamic_users.mat'

# # load them
# base  = scipy.io.loadmat(base_path)
# trans = scipy.io.loadmat(trans_path)

# # keys to ignore
# ignore = {'__header__','__version__','__globals__','input_h'}

# # find common, non‐ignored fields
# common = [k for k in base.keys()
#           if k in trans.keys() and k not in ignore]

# # print header
# print(f"{'Field':<20} {'Base Total':>15}    {'Transformer Total':>15}")
# print(f"{'-'*20} {'-'*15}    {'-'*15}")

# # sum and print each
# for field in common:
#     b = np.array(base[field]).flatten()
#     t = np.array(trans[field]).flatten()
#     print(f"{field:<20} {b.sum():15.4f}    {t.sum():15.4f}")

import scipy.io
import numpy as np

# Load your two .mat files
base = scipy.io.loadmat('result_base_dynamic_users.mat')
trans = scipy.io.loadmat('result_transformer_dynamic_users.mat')

# Fields to ignore
ignore = {'__header__', '__version__', '__globals__', 'input_h'}

# Find all common keys except the ignored ones
common = [k for k in base if k in trans and k not in ignore]

# Also explicitly look for any “loss” fields
loss_fields = [k for k in base if 'loss' in k.lower() and k in trans]
for k in loss_fields:
    if k not in common:
        common.append(k)

# Print header
print(f"{'Field':<20} {'Base Total':>15} {'Transformer Total':>20}")
print(f"{'-'*20} {'-'*15} {'-'*20}")

# Sum & print
for field in common:
    b = np.array(base[field]).flatten()
    t = np.array(trans[field]).flatten()
    # Some fields might be strings or cells—skip non-numeric
    if not np.issubdtype(b.dtype, np.number) or not np.issubdtype(t.dtype, np.number):
        continue
    print(f"{field:<20} {np.nansum(b):15.4f} {np.nansum(t):20.4f}")
