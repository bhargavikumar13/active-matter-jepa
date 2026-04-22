import h5py
import os
import sys
import numpy as np

data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars('/scratch/$USER/data/active_matter/data')
print("Data dir:", data_dir)
print("Splits:", os.listdir(data_dir))

train_dir = os.path.join(data_dir, 'train')
train_files = sorted(os.listdir(train_dir))
print(f"\nNumber of training files: {len(train_files)}")

fpath = os.path.join(train_dir, train_files[0])

def print_hdf5_item(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  [Dataset] {name}: shape={obj.shape}, dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"  [Group]   {name}/")

def summarize(obj, indent=0):
    prefix = "  " * indent
    for k, v in obj.items():
        if isinstance(v, h5py.Dataset):
            val = v[()] if v.shape == () else v.shape
            print(f"{prefix}  {k}: {val}")
        elif isinstance(v, h5py.Group):
            print(f"{prefix}  {k}/")
            summarize(v, indent + 1)

with h5py.File(fpath, 'r') as f:
    print(f"\nHDF5 structure in: {train_files[0]}")
    f.visititems(print_hdf5_item)

    print(f"\nPhysical parameters (scalars):")
    for k, v in f['scalars'].items():
        print(f"  {k} = {v[()]}")

    print(f"\nField snapshots:")
    for field_key in ['t0_fields', 't1_fields', 't2_fields']:
        obj = f[field_key]
        if isinstance(obj, h5py.Group):
            print(f"  {field_key}/")
            for k, v in obj.items():
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {field_key}: shape={obj.shape}, dtype={obj.dtype}")

    print(f"\nBoundary conditions / dimensions:")
    for key in ['boundary_conditions', 'dimensions']:
        if key in f:
            print(f"  {key}/")
            summarize(f[key], indent=1)
