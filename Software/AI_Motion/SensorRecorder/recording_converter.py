import os
import glob
import pickle
import numpy as np

def convert_recording(old_filepath, output_dir):
    # Extract the base filename
    basename = os.path.basename(old_filepath)
    
    # Load the old pickle data
    with open(old_filepath, 'rb') as f:
        old_data = pickle.load(f)
        
    # Extract lists from the old dictionary
    time_stamps = old_data.get("time_stamps", [])
    sensor_ids = old_data.get("sensor_ids", [])
    sensor_values = old_data.get("sensor_values", [])
    
    # Group data by sensor ID (OSC address) to match the new format
    grouped_data = {}
    for t, s_id, v in zip(time_stamps, sensor_ids, sensor_values):
        if s_id not in grouped_data:
            grouped_data[s_id] = {"timestamps": [], "values": []}
        
        grouped_data[s_id]["timestamps"].append(t)
        grouped_data[s_id]["values"].append(v)
        
    # Prepare the dictionary for np.savez_compressed
    save_dict = {}
    for addr, d in grouped_data.items():
        # Match the datatypes specified in the new recorder script
        save_dict[addr + "_timestamps"] = np.array(d["timestamps"], dtype=np.float64)
        save_dict[addr + "_values"] = np.array(d["values"], dtype=np.float32)
        
    # Generate new filename by replacing .pkl with .npz
    new_filename = basename.replace(".pkl", ".npz")
    new_filepath = os.path.join(output_dir, new_filename)
    
    # Save as compressed NumPy archive
    np.savez_compressed(new_filepath, **save_dict)
    print(f"Converted: {basename}\n       -> {new_filename}\n")

def main():
    input_dir = "recordings_old"  # Folder containing the old .pkl files
    output_dir = "recordings_new" 
    
    if not os.path.exists(input_dir):
        print(f"Directory '{input_dir}' not found. Please place this script next to your recordings folder.")
        return
        
    old_files = glob.glob(os.path.join(input_dir, "*.pkl"))
    
    if not old_files:
        print(f"No .pkl files found in '{input_dir}'.")
        return
        
    print(f"Found {len(old_files)} old recordings. Starting conversion...\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for file_path in old_files:
        try:
            convert_recording(file_path, output_dir)
        except Exception as e:
            print(f"Failed to convert {file_path}: {e}")
            
    print("Conversion complete!")

if __name__ == "__main__":
    main()