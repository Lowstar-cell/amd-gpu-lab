import pandas as pd
import sys

def inspect_h5_file(path):
    """
    Tries to read an HDF5 file with pandas and prints its structure.
    """
    print(f"--- Inspecting HDF5 file: {path} ---")
    try:
        with pd.HDFStore(path, mode='r') as store:
            if not store.keys():
                print("HDF5 file is empty or contains no keys.")
                return

            print(f"Keys found in file: {store.keys()}")
            
            # Try to read and print the head of the first key
            first_key = store.keys()[0]
            print(f"--- Reading first 5 rows from key: '{first_key}' ---")
            df = store.get(first_key)
            print(df.head())
            print("\n--- Data columns and types ---")
            df.info()

    except ImportError:
        print("\n[Error] The 'tables' library is required to read HDF5 files with pandas, but it's not installed.")
        print("You can try installing it in your virtual environment with: pip install tables")
    except Exception as e:
        print(f"\n[Error] An error occurred while reading the file: {e}")

if __name__ == "__main__":
    import glob
    # Find all HDF5 files in the directory
    data_dir = "gpu_rasterization/data/"
    files_to_inspect = glob.glob(data_dir + "*.h5")
    
    if not files_to_inspect:
        print(f"No .h5 files found in {data_dir}")
    else:
        print(f"Found {len(files_to_inspect)} files to inspect...")
        for file_path in files_to_inspect:
            inspect_h5_file(file_path)
            print("-" * 50)
