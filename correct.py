import sys
import numpy as np
import cv2
import math
import os
import glob

# --- Constants ---
THRESHOLD_RATIO = 2000
MIN_AVG_RED = 60
# MAX_HUE_SHIFT = 120 # Original Value
MAX_HUE_SHIFT = 100 # Reduced max hue shift slightly
BLUE_MAGIC_VALUE = 1.2
SAMPLE_SECONDS = 2 # Extracts color correction from every N seconds

def hue_shift_red(mat, h):

    U = math.cos(h * math.pi / 180)
    W = math.sin(h * math.pi / 180)

    r = (0.299 + 0.701 * U + 0.168 * W) * mat[..., 0]
    g = (0.587 - 0.587 * U + 0.330 * W) * mat[..., 1]
    b = (0.114 - 0.114 * U - 0.497 * W) * mat[..., 2]

    return np.dstack([r, g, b])

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

# --- New constants to control brightness/gain ---
MAX_NORMALIZATION_GAIN = 3.5 # Clamp maximum gain applied during normalization
MIN_NORM_RANGE = 50         # Ensure the normalization range is at least this wide

# --- Optional Gamma Correction ---
# APPLY_GAMMA_CORRECTION = True
# GAMMA_VALUE = 1.1 # > 1 compresses highlights, < 1 expands them. 1.1 is subtle.

# (Keep hue_shift_red, normalizing_interval as they are, unless normalizing_interval fix was needed)
# Make sure the normalizing_interval fix from the previous step is included if it wasn't already:
def normalizing_interval(array):

    high = 255
    low = 0
    max_dist = 0

    # Find the first non-zero value as a potential start for 'low'
    first_valid_index = -1
    for i in range(len(array)):
        if array[i] != 0:
            first_valid_index = i
            break

    # If all values are 0 (unlikely but possible), return default
    if first_valid_index == -1:
        # Fallback to a safe default range if no valid data points
        return (0, 255)

    low = array[first_valid_index] # Initialize low

    # Iterate from the second valid value onwards
    current_low_candidate = low
    for i in range(first_valid_index + 1, len(array)):
        if array[i] == 0: # Skip zero placeholders
            continue

        dist = array[i] - current_low_candidate
        if(dist > max_dist):
            max_dist = dist
            high = array[i]
            low = current_low_candidate

        current_low_candidate = array[i] # Update for next comparison

    # Handle edge case: if only one non-zero value found or high didn't move
    if high == 255 and low == array[first_valid_index] and max_dist == 0:
         # If the single valid point isn't 0, try using 0 as low
         if low > 0 :
             low = 0 # Widen the range by setting low to 0
         # else keep low=0, high=255

    # Ensure low and high are different if possible, handle plateau at 255
    if low >= high:
        # Try to establish a minimal valid range if they ended up the same
        if high < 255:
            high += 1
        elif low > 0:
            low -= 1
        else: # Failsafe if low=high=0 or low=high=255
            low = 0
            high = 255

    return (low, high)


def apply_filter(mat, filt):
    # Ensure input is float for calculations
    mat_float = mat.astype(np.float32)

    r = mat_float[..., 0]
    g = mat_float[..., 1]
    b = mat_float[..., 2]

    # Apply filter matrix operations
    new_r = r * filt[0] + g*filt[1] + b*filt[2] + filt[4]*255
    new_g = r * filt[5] + g*filt[6] + b*filt[7] + filt[9]*255 # Corrected indices for green channel
    new_b = r * filt[10] + g*filt[11] + b*filt[12] + filt[14]*255 # Corrected indices for blue channel

    # Stack, clip, and convert back to uint8
    filtered_mat = np.dstack([new_r, new_g, new_b])

    # --- Optional Gamma Correction Step ---
    # if APPLY_GAMMA_CORRECTION:
    #     # Avoid division by zero or log(0) errors
    #     filtered_mat = np.clip(filtered_mat, 1, 255) # Clip slightly above 0 before division
    #     filtered_mat = np.power(filtered_mat / 255.0, 1.0 / GAMMA_VALUE) * 255.0

    # Final Clipping
    filtered_mat = np.clip(filtered_mat, 0, 255).astype(np.uint8)

    return filtered_mat


# --- Function for aggregate statistics ---
def get_aggregate_filter_matrix(image_paths):
    # (Keep the analysis part the same as before)
    # ... (analysis loop accumulating sum_avg_r/g/b, sum_hist_r/g/b, total_pixels, valid_image_count) ...
    print(f"Analyzing {len(image_paths)} images for aggregate statistics...")
    if not image_paths:
        print("Warning: No image paths provided for aggregate analysis.")
        return np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32) # Identity

    sum_avg_r, sum_avg_g, sum_avg_b = 0.0, 0.0, 0.0
    sum_hist_r = np.zeros((256, 1), dtype=np.float32)
    sum_hist_g = np.zeros((256, 1), dtype=np.float32)
    sum_hist_b = np.zeros((256, 1), dtype=np.float32)
    total_pixels = 0
    valid_image_count = 0

    for i, img_path in enumerate(image_paths):
        print(f"  Analyzing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}", end='\r')
        mat = cv2.imread(img_path)
        if mat is None:
            print(f"\nWarning: Could not read image {img_path}. Skipping.")
            continue

        mat_rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        mat_resized = cv2.resize(mat_rgb, (256, 256))

        avg_bgr = cv2.mean(mat_resized)[:3]
        sum_avg_b += avg_bgr[0]
        sum_avg_g += avg_bgr[1]
        sum_avg_r += avg_bgr[2]

        hist_r = cv2.calcHist([mat_resized], [0], None, [256], [0,256])
        hist_g = cv2.calcHist([mat_resized], [1], None, [256], [0,256])
        hist_b = cv2.calcHist([mat_resized], [2], None, [256], [0,256])
        sum_hist_r += hist_r
        sum_hist_g += hist_g
        sum_hist_b += hist_b

        total_pixels += mat_resized.shape[0] * mat_resized.shape[1]
        valid_image_count += 1

    print("\nAggregate analysis complete.")

    if valid_image_count == 0:
        print("Error: No valid images found in the directory.")
        return np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32) # Identity

    overall_avg_r = sum_avg_r / valid_image_count
    overall_avg_g = sum_avg_g / valid_image_count
    overall_avg_b = sum_avg_b / valid_image_count
    avg_mat = np.array([[[overall_avg_r, overall_avg_g, overall_avg_b]]], dtype=np.float32)

    # Find hue shift
    hue_shift = 0
    current_avg_pixel = avg_mat.copy()
    new_avg_r_component = current_avg_pixel[0,0,0]

    while(new_avg_r_component < MIN_AVG_RED):
        hue_shift += 1
        if hue_shift > MAX_HUE_SHIFT: # Use the reduced MAX_HUE_SHIFT
            print(f"Warning: Max hue shift ({MAX_HUE_SHIFT}) reached during aggregate analysis.")
            hue_shift = MAX_HUE_SHIFT
            break
        shifted_avg_pixel = hue_shift_red(current_avg_pixel, hue_shift)
        new_avg_r_component = shifted_avg_pixel[0,0,0]
    print(f"Calculated aggregate hue shift: {hue_shift}")

    # Use aggregate histograms for normalization
    normalize_mat = np.zeros((256, 3))
    threshold_level = total_pixels / THRESHOLD_RATIO
    print(f"Aggregate pixel count: {total_pixels}, Threshold level: {threshold_level}")

    for x in range(256):
        if sum_hist_r[x] < threshold_level: normalize_mat[x][0] = x
        if sum_hist_g[x] < threshold_level: normalize_mat[x][1] = x
        if sum_hist_b[x] < threshold_level: normalize_mat[x][2] = x

    normalize_mat[255][0] = 255
    normalize_mat[255][1] = 255
    normalize_mat[255][2] = 255

    adjust_r_low, adjust_r_high = normalizing_interval(normalize_mat[..., 0])
    adjust_g_low, adjust_g_high = normalizing_interval(normalize_mat[..., 1])
    adjust_b_low, adjust_b_high = normalizing_interval(normalize_mat[..., 2])

    print(f"Initial Normalization R: ({adjust_r_low}, {adjust_r_high})")
    print(f"Initial Normalization G: ({adjust_g_low}, {adjust_g_high})")
    print(f"Initial Normalization B: ({adjust_b_low}, {adjust_b_high})")

    # --- Add Minimum Normalization Range ---
    adjust_r_high = max(adjust_r_high, adjust_r_low + MIN_NORM_RANGE)
    adjust_g_high = max(adjust_g_high, adjust_g_low + MIN_NORM_RANGE)
    adjust_b_high = max(adjust_b_high, adjust_b_low + MIN_NORM_RANGE)
    # Ensure high does not exceed 255 after adding range
    adjust_r_high = min(adjust_r_high, 255)
    adjust_g_high = min(adjust_g_high, 255)
    adjust_b_high = min(adjust_b_high, 255)
    # Ensure low is still less than high after adjustments
    adjust_r_low = min(adjust_r_low, adjust_r_high -1)
    adjust_g_low = min(adjust_g_low, adjust_g_high -1)
    adjust_b_low = min(adjust_b_low, adjust_b_high -1)

    print(f"Adjusted Normalization R (Min Range {MIN_NORM_RANGE}): ({adjust_r_low}, {adjust_r_high})")
    print(f"Adjusted Normalization G (Min Range {MIN_NORM_RANGE}): ({adjust_g_low}, {adjust_g_high})")
    print(f"Adjusted Normalization B (Min Range {MIN_NORM_RANGE}): ({adjust_b_low}, {adjust_b_high})")


    # Calculate filter matrix components
    shifted = hue_shift_red(np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32), hue_shift)
    shifted_r, shifted_g, shifted_b = shifted[0][0]

    # Calculate gains
    red_gain_norm = 255.0 / (adjust_r_high - adjust_r_low)
    green_gain_norm = 255.0 / (adjust_g_high - adjust_g_low)
    blue_gain_norm = 255.0 / (adjust_b_high - adjust_b_low)

    # --- Clamp Maximum Gain ---
    red_gain_norm = min(red_gain_norm, MAX_NORMALIZATION_GAIN)
    green_gain_norm = min(green_gain_norm, MAX_NORMALIZATION_GAIN)
    blue_gain_norm = min(blue_gain_norm, MAX_NORMALIZATION_GAIN)
    print(f"Clamped Gains (Max {MAX_NORMALIZATION_GAIN}): R={red_gain_norm:.2f}, G={green_gain_norm:.2f}, B={blue_gain_norm:.2f}")


    # Calculate offsets based on normalization low points
    redOffset = (-adjust_r_low / 255.0) * red_gain_norm
    greenOffset = (-adjust_g_low / 255.0) * green_gain_norm
    blueOffset = (-adjust_b_low / 255.0) * blue_gain_norm

    # Combine hue shift factors with red normalization gain
    adjust_red = shifted_r * red_gain_norm
    adjust_red_green = shifted_g * red_gain_norm
    adjust_red_blue = shifted_b * red_gain_norm * BLUE_MAGIC_VALUE

    filter_matrix = np.array([
        adjust_red, adjust_red_green, adjust_red_blue, 0, redOffset,
        0, green_gain_norm, 0, 0, greenOffset,
        0, 0, blue_gain_norm, 0, blueOffset,
        0, 0, 0, 1, 0,
    ], dtype=np.float32)

    print("Calculated Aggregate Filter Matrix (Clamped):\n", filter_matrix[:15].reshape(3,5))

    return filter_matrix


# --- Function to process a directory (remains the same logic, uses the modified get_aggregate_filter_matrix) ---
def correct_directory(input_dir, output_dir, output_prefix, yield_preview=False):
    # ... (same as before) ...
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))

    if not image_paths:
        yield "Error: No image files found in the selected directory.", 0, 0, None
        return

    os.makedirs(output_dir, exist_ok=True)

    yield "Status: Analyzing directory for aggregate filter...", 0, len(image_paths), None
    try:
        aggregate_filter_matrix = get_aggregate_filter_matrix(image_paths) # Calls the modified function
    except Exception as e:
         yield f"Error during analysis: {e}", 0, len(image_paths), None
         return

    total_images = len(image_paths)
    for i, img_path in enumerate(image_paths):
        base_name = os.path.basename(img_path)
        output_name = f"{output_prefix}_{base_name}"
        output_path = os.path.join(output_dir, output_name)
        status = f"Status: Applying filter ({i+1}/{total_images}) to {base_name}..."
        preview_bytes = None
        try:
            original_mat = cv2.imread(img_path)
            if original_mat is None:
                print(f"\nWarning: Could not read image {img_path} during application phase. Skipping.")
                yield status + " Skipped (Read Error).", i + 1, total_images, None
                continue
            rgb_mat = cv2.cvtColor(original_mat, cv2.COLOR_BGR2RGB)
            corrected_mat_rgb = apply_filter(rgb_mat, aggregate_filter_matrix) # Applies the clamped filter
            corrected_mat_bgr = cv2.cvtColor(corrected_mat_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, corrected_mat_bgr)

            if yield_preview:
                preview = original_mat.copy()
                width = preview.shape[1] // 2
                if corrected_mat_bgr.shape == original_mat.shape:
                     preview[::, width:] = corrected_mat_bgr[::, width:]
                else:
                     preview = corrected_mat_bgr
                preview_resized = cv2.resize(preview, (480, 270))
                ret, png = cv2.imencode('.png', preview_resized)
                if ret: preview_bytes = png.tobytes()

            yield status, i + 1, total_images, preview_bytes
        except Exception as e:
            error_msg = f"\nError processing {base_name}: {e}"
            print(error_msg)
            yield status + f" Error ({e})", i + 1, total_images, None
    yield f"Status: Finished processing {total_images} images.", total_images, total_images, None


# --- Original single image correction (modified to include clamping) ---
def get_filter_matrix(mat): # For single image / video frames
    mat_resized = cv2.resize(mat, (256, 256))
    avg_mat_bgr = cv2.mean(mat_resized)[:3]
    avg_pixel_bgr = np.array([[[avg_mat_bgr[0], avg_mat_bgr[1], avg_mat_bgr[2]]]], dtype=np.float32)
    avg_pixel_rgb = cv2.cvtColor(avg_pixel_bgr, cv2.COLOR_BGR2RGB)

    hue_shift = 0
    current_avg_pixel = avg_pixel_rgb.copy()
    new_avg_r_component = current_avg_pixel[0,0,0]

    while(new_avg_r_component < MIN_AVG_RED):
        hue_shift += 1
        if hue_shift > MAX_HUE_SHIFT: # Use reduced MAX_HUE_SHIFT
             # print(f"Warning: Max hue shift ({MAX_HUE_SHIFT}) reached for single frame.") # Optional print
             hue_shift = MAX_HUE_SHIFT
             break
        shifted_avg_pixel = hue_shift_red(current_avg_pixel, hue_shift)
        new_avg_r_component = shifted_avg_pixel[0,0,0]

    # Apply hue shift to whole image's red channel for histogram base
    shifted_mat = hue_shift_red(mat_resized.astype(np.float32), hue_shift)
    new_r_channel = np.sum(shifted_mat, axis=2)
    new_r_channel = np.clip(new_r_channel, 0, 255)
    mat_for_hist = mat_resized.copy()
    mat_for_hist[..., 0] = new_r_channel.astype(np.uint8)

    hist_r = cv2.calcHist([mat_for_hist], [0], None, [256], [0,256])
    hist_g = cv2.calcHist([mat_resized], [1], None, [256], [0,256]) # Use original G,B hist
    hist_b = cv2.calcHist([mat_resized], [2], None, [256], [0,256])

    normalize_mat = np.zeros((256, 3))
    threshold_level = (mat_resized.shape[0]*mat_resized.shape[1])/THRESHOLD_RATIO
    for x in range(256):
        if hist_r[x] < threshold_level: normalize_mat[x][0] = x
        if hist_g[x] < threshold_level: normalize_mat[x][1] = x
        if hist_b[x] < threshold_level: normalize_mat[x][2] = x

    normalize_mat[255][0] = 255
    normalize_mat[255][1] = 255
    normalize_mat[255][2] = 255

    adjust_r_low, adjust_r_high = normalizing_interval(normalize_mat[..., 0])
    adjust_g_low, adjust_g_high = normalizing_interval(normalize_mat[..., 1])
    adjust_b_low, adjust_b_high = normalizing_interval(normalize_mat[..., 2])

    # --- Add Minimum Normalization Range ---
    adjust_r_high = max(adjust_r_high, adjust_r_low + MIN_NORM_RANGE)
    adjust_g_high = max(adjust_g_high, adjust_g_low + MIN_NORM_RANGE)
    adjust_b_high = max(adjust_b_high, adjust_b_low + MIN_NORM_RANGE)
    adjust_r_high = min(adjust_r_high, 255)
    adjust_g_high = min(adjust_g_high, 255)
    adjust_b_high = min(adjust_b_high, 255)
    adjust_r_low = min(adjust_r_low, adjust_r_high -1)
    adjust_g_low = min(adjust_g_low, adjust_g_high -1)
    adjust_b_low = min(adjust_b_low, adjust_b_high -1)


    shifted = hue_shift_red(np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32), hue_shift)
    shifted_r, shifted_g, shifted_b = shifted[0][0]

    red_gain_norm = 255.0 / (adjust_r_high - adjust_r_low)
    green_gain_norm = 255.0 / (adjust_g_high - adjust_g_low)
    blue_gain_norm = 255.0 / (adjust_b_high - adjust_b_low)

    # --- Clamp Maximum Gain ---
    red_gain_norm = min(red_gain_norm, MAX_NORMALIZATION_GAIN)
    green_gain_norm = min(green_gain_norm, MAX_NORMALIZATION_GAIN)
    blue_gain_norm = min(blue_gain_norm, MAX_NORMALIZATION_GAIN)

    redOffset = (-adjust_r_low / 255.0) * red_gain_norm
    greenOffset = (-adjust_g_low / 255.0) * green_gain_norm
    blueOffset = (-adjust_b_low / 255.0) * blue_gain_norm

    adjust_red = shifted_r * red_gain_norm
    adjust_red_green = shifted_g * red_gain_norm
    adjust_red_blue = shifted_b * red_gain_norm * BLUE_MAGIC_VALUE

    filter_matrix = np.array([
        adjust_red, adjust_red_green, adjust_red_blue, 0, redOffset,
        0, green_gain_norm, 0, 0, greenOffset,
        0, 0, blue_gain_norm, 0, blueOffset,
        0, 0, 0, 1, 0,
    ], dtype=np.float32)

    return filter_matrix


def correct(mat): # Corrects a single matrix using its own stats
    original_mat = mat.copy() # Operate on RGB

    filter_matrix = get_filter_matrix(original_mat) # Get filter based *only* on this matrix

    corrected_mat = apply_filter(original_mat, filter_matrix) # Apply filter (expects RGB)
    # corrected_mat is RGB, convert back to BGR only when saving/displaying with OpenCV

    return corrected_mat # Return RGB

def correct_image(input_path, output_path): # Corrects single image file
    mat = cv2.imread(input_path)
    if mat is None:
        print(f"Error reading image {input_path}")
        return None
    rgb_mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)

    corrected_rgb_mat = correct(rgb_mat) # Returns RGB

    corrected_bgr_mat = cv2.cvtColor(corrected_rgb_mat, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, corrected_bgr_mat)

    # Preview generation
    preview = mat.copy() # Original BGR
    width = preview.shape[1] // 2
    if corrected_bgr_mat.shape == preview.shape:
        preview[::, width:] = corrected_bgr_mat[::, width:]

    preview_resized = cv2.resize(preview, (480, 270)) # Smaller preview
    ret, png = cv2.imencode('.png', preview_resized)
    if ret:
        return png.tobytes()
    else:
        return None


# --- Video functions remain largely unchanged, but use the single-frame `get_filter_matrix` ---
def analyze_video(input_video_path, output_video_path):

    # Initialize new video writer
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        # Yield a dictionary indicating failure or raise an error
        yield {"error": f"Could not open video {input_video_path}"}
        return

    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    frame_count = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        print(f"Warning: Invalid FPS ({fps}) or Frame Count ({frame_count}) for {input_video_path}")
        # Attempt to proceed but might cause issues later
        fps = 30 if fps <= 0 else fps # Default fps
        # Frame count is harder to guess, interpolation might fail

    # Get filter matrices for every Nth second
    filter_matrix_indexes = []
    filter_matrices = []
    count = 0

    print("Analyzing video...")
    while(cap.isOpened()):

        count += 1
        ret, frame = cap.read()
        if not ret:
            print(f"End of video or read error at frame {count}.")
             # Check if we reached expected end, otherwise it's an early termination
            if count < frame_count * 0.95: # Allow some tolerance
                print(f"Warning: Video ended earlier than expected ({count}/{frame_count} frames).")
            break # Normal or error exit

        # Failsafe (optional, frame_count should handle most cases)
        # if count >= 1e6: break

        # Pick filter matrix from every N seconds (ensure SAMPLE_SECONDS >= 1)
        sample_interval = max(1, fps * SAMPLE_SECONDS)
        if count % sample_interval == 0:
            mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
             # Use the original single-frame filter logic for videos
            try:
                 fm = get_filter_matrix(mat)
                 filter_matrix_indexes.append(count)
                 filter_matrices.append(fm)
                 print(f"Analyzed frame {count} for filter...", end='\r')
            except Exception as e:
                 print(f"\nError calculating filter for frame {count}: {e}")
                 # Skip this frame's filter? Or stop analysis? For now, skip.
                 continue


        yield count # Yield progress (frame number)

    cap.release()
    print("\nVideo analysis finished.")

    if not filter_matrices:
         print("Error: No filter matrices were generated during video analysis.")
         yield {"error": "No filters generated. Video might be too short or analysis failed."}
         return


    # Build a interpolation function to get filter matrix at any given frame
    filter_matrices = np.array(filter_matrices)
    frame_count = count # Use actual processed count

    yield {
        "input_video_path": input_video_path,
        "output_video_path": output_video_path,
        "fps": fps,
        "frame_count": frame_count, # Use actual count
        "filters": filter_matrices,
        "filter_indices": filter_matrix_indexes
    }

def process_video(video_data, yield_preview=False):

    if "error" in video_data:
         print(f"Skipping video processing due to analysis error: {video_data['error']}")
         return

    cap = cv2.VideoCapture(video_data["input_video_path"])
    if not cap.isOpened():
         print(f"Error opening video file for processing: {video_data['input_video_path']}")
         return


    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_data["fps"]
    output_path = video_data["output_video_path"]
    frame_count = video_data["frame_count"] # Use count from analysis

    # Ensure valid dimensions and fps
    if not (frame_width > 0 and frame_height > 0 and fps > 0):
         print(f"Error: Invalid video properties for output W:{frame_width}, H:{frame_height}, FPS:{fps}")
         cap.release()
         return


    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or use another codec if needed
    new_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not new_video.isOpened():
         print(f"Error: Could not create video writer for {output_path}")
         cap.release()
         return

    filter_matrices = video_data["filters"]
    filter_indices = video_data["filter_indices"]

    if not filter_indices: # Should have been caught in analysis, but double-check
        print("Error: No filter indices found for interpolation.")
        cap.release()
        new_video.release()
        # Clean up potentially created empty file
        if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
            os.remove(output_path)
        return

    filter_matrix_size = len(filter_matrices[0])

    # Add bounds_error=False and fill_value to handle extrapolation
    # Use first filter before first index, last filter after last index
    def get_interpolated_filter_matrix(frame_number):
        interpolated_filter = []
        for x in range(filter_matrix_size):
             interp_val = np.interp(frame_number,
                                   filter_indices,
                                   filter_matrices[..., x],
                                   left=filter_matrices[0, x], # Use first filter value if before range
                                   right=filter_matrices[-1, x]) # Use last filter value if after range
             interpolated_filter.append(interp_val)
        return interpolated_filter


    print("Processing video...")
    count = 0
    while(cap.isOpened()):

        count += 1
        ret, frame = cap.read()

        if not ret:
            print(f"End of video or read error during processing at frame {count}.")
            if count < frame_count * 0.95:
                print(f"Warning: Processing ended earlier than expected ({count}/{frame_count} frames).")
            break

        # Stop if we exceed the analyzed frame count (prevents infinite loops on weird videos)
        if count > frame_count:
            print(f"Stopping processing as frame count ({count}) exceeds analyzed count ({frame_count}).")
            break


        percent = 100 * count / frame_count if frame_count > 0 else 0
        # print("{:.2f}".format(percent), end=" % \r") # Console output handled by GUI now

        # Apply the filter
        rgb_mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        interpolated_filter_matrix = get_interpolated_filter_matrix(count)
        corrected_rgb_mat = apply_filter(rgb_mat, interpolated_filter_matrix)
        corrected_bgr_mat = cv2.cvtColor(corrected_rgb_mat, cv2.COLOR_RGB2BGR)

        new_video.write(corrected_bgr_mat)

        preview_bytes = None
        if yield_preview:
            try:
                preview = frame.copy()
                width = preview.shape[1] // 2
                height = preview.shape[0] // 2 # Keep aspect ratio for resize
                preview_width_out = 480
                preview_height_out = int(preview_width_out * (height/width)) if width > 0 else 270


                if corrected_bgr_mat.shape == preview.shape:
                     preview[::, width:] = corrected_bgr_mat[::, width:]
                else:
                     preview = corrected_bgr_mat # Show only corrected if shapes mismatch

                preview_resized = cv2.resize(preview, (preview_width_out, preview_height_out))
                ret_enc, png = cv2.imencode('.png', preview_resized)
                if ret_enc:
                    preview_bytes = png.tobytes()
            except Exception as e:
                print(f"Error generating preview for frame {count}: {e}")
                preview_bytes = None # Ensure it's None on error


        yield percent, preview_bytes # Yield progress and preview bytes


    print("\nVideo processing finished.")
    cap.release()
    new_video.release()


if __name__ == "__main__":

    if len(sys.argv) < 4: # Adjusted for new directory mode
        print("Usage")
        print("-"*20)
        print("For single image:")
        print("$python correct.py image <source_image_path> <output_image_path>\n")
        print("-"*20)
        print("For video:")
        print("$python correct.py video <source_video_path> <output_video_path>\n")
        print("-"*20)
        print("For directory:")
        print("$python correct.py directory <source_directory> <output_directory> [prefix]\n")
        exit(0)

    mode = sys.argv[1].lower()
    input_path = sys.argv[2]
    output_path = sys.argv[3]


    if mode == "image":
        if not os.path.isfile(input_path):
             print(f"Error: Input image not found: {input_path}")
             exit(1)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Correcting single image: {input_path} -> {output_path}")
        correct_image(input_path, output_path)
        print("Done.")

    elif mode == "video":
         if not os.path.isfile(input_path):
             print(f"Error: Input video not found: {input_path}")
             exit(1)
         os.makedirs(os.path.dirname(output_path), exist_ok=True)
         print(f"Analyzing video: {input_path}")
         video_data = None
         for item in analyze_video(input_path, output_path):
             if isinstance(item, dict):
                 video_data = item
                 if "error" in video_data:
                     print(f"Analysis failed: {video_data['error']}")
                     exit(1)
                 break # Got the data dict
             elif isinstance(item, int):
                  print(f"Analyzed frame {item}", end='\r') # Show progress
             else:
                  print("Unexpected item from analyze_video:", item)


         if video_data:
             print(f"\nProcessing video: {input_path} -> {output_path}")
             # Consume the generator to run the process
             for percent, _ in process_video(video_data, yield_preview=False):
                  if percent is not None:
                       print(f"Processing: {percent:.2f}%", end='\r')
             print("\nVideo processing complete.")
         else:
              print("Error: Did not receive video data after analysis.")
              exit(1)

    elif mode == "directory":
        if not os.path.isdir(input_path):
            print(f"Error: Input directory not found: {input_path}")
            exit(1)
        if not os.path.isdir(output_path):
             print(f"Creating output directory: {output_path}")
             os.makedirs(output_path, exist_ok=True)

        prefix = sys.argv[4] if len(sys.argv) > 4 else "corrected"
        print(f"Correcting directory: {input_path} -> {output_path} (prefix: {prefix})")

        # Consume the generator to run the process
        for status, current, total, _ in correct_directory(input_path, output_path, prefix, yield_preview=False):
             print(status, end='\r') # Show progress from generator

        print("\nDirectory processing complete.")

    else:
        print(f"Error: Unknown mode '{mode}'. Use 'image', 'video', or 'directory'.")
        exit(1)