import argparse
import numpy as np
import os
import sys

def identify_timestamp_keys(data_dict):
    """
    Identifies which keys in the dictionary represent timestamp arrays.
    It relies on structural clues rather than hardcoded names:
      1. Must be 1-dimensional and numeric.
      2. Must be non-decreasing (monotonic).
      3. Often has 'time' or 'ts' in the key name (as a fast-path).
      4. If names are completely obscure, it pairs up with another array of the 
         same length (the sensor values) that is multidimensional or non-monotonic.
    """
    ts_keys = set()
    
    for key, arr in data_dict.items():
        # Condition 1: 1D and numeric
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            k_lower = key.lower()
            
            # Fast-path: obvious naming convention
            if 'time' in k_lower or 'ts' in k_lower or k_lower.endswith('_t'):
                ts_keys.add(key)
                continue
            
            # Structural fallback: non-decreasing and pairs with a data array
            if len(arr) > 1 and np.all(np.diff(arr) >= 0):
                for other_key, other_arr in data_dict.items():
                    if other_key != key and other_arr.shape[0] == arr.shape[0]:
                        # If the paired array is multi-dimensional or not monotonically increasing, 
                        # it's the value array, making our current 'arr' the timestamp array.
                        if other_arr.ndim > 1 or not np.all(np.diff(other_arr) >= 0):
                            ts_keys.add(key)
                            break
                            
    return ts_keys

def main():
    parser = argparse.ArgumentParser(
        description="Rewrite NPZ timestamps assuming a fixed framerate, aligning all sensors."
    )
    parser.add_argument("input_npz", type=str, help="Path to the input .npz file")
    parser.add_argument("fps", type=float, help="Target framerate (e.g., 30.0, 50.0)")
    parser.add_argument("output_npz", type=str, help="Path to save the output .npz file")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_npz):
        print(f"Error: Input file '{args.input_npz}' not found.")
        sys.exit(1)

    print(f"Loading '{args.input_npz}'...")
    
    # Load data into a standard dictionary
    with np.load(args.input_npz) as data:
        data_dict = {key: data[key] for key in data.files}

    ts_keys = identify_timestamp_keys(data_dict)
    
    if not ts_keys:
        print("Warning: Could not automatically identify any timestamp arrays based on structure.")
    
    out_dict = {}
    
    for key, arr in data_dict.items():
        if key in ts_keys:
            print(f" -> Replacing timestamps for: '{key}' (length {len(arr)})")
            
            # Generate new timestamps: [0.0, 1/fps, 2/fps, ..., (N-1)/fps]
            # This guarantees that index `i` is identical across all sensor arrays
            new_ts = (np.arange(len(arr), dtype=np.float32) / args.fps)
            out_dict[key] = new_ts
        else:
            # Keep sensor value arrays exactly as they are
            out_dict[key] = arr

    print(f"Saving modified data to '{args.output_npz}'...")
    np.savez(args.output_npz, **out_dict)
    print("Done!")

if __name__ == "__main__":
    main()