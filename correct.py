# /correct.py:
#--------------------------------------------------------------------------------
import sys
import numpy as np
import cv2
import math
import os # Added
import glob # Added

THRESHOLD_RATIO = 2000
MIN_AVG_RED = 60
MAX_HUE_SHIFT = 120
BLUE_MAGIC_VALUE = 1.2
SAMPLE_SECONDS = 2 # Extracts color correction from every N seconds

# Supported image extensions for directory processing
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff') # Added

def hue_shift_red(mat, h):

    U = math.cos(h * math.pi / 180)
    W = math.sin(h * math.pi / 180)

    r = (0.299 + 0.701 * U + 0.168 * W) * mat[..., 0]
    g = (0.587 - 0.587 * U + 0.330 * W) * mat[..., 1]
    b = (0.114 - 0.114 * U - 0.497 * W) * mat[..., 2]

    return np.dstack([r, g, b])

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

    # Handle edge case: if only one non-zero value found
    if high == 255 and low == array[first_valid_index] and max_dist == 0:
         if first_valid_index > 0: # If the first value wasn't 0 itself
              low = array[first_valid_index-1] # Use the value before it if possible (should be 0)
         #else keep low as the first value. High remains 255.

    # Ensure low and high are different if possible, handle plateau at 255
    if low == high:
       if high < 255:
           high += 1 # Ensure at least a minimal interval if stuck
       elif low > 0:
           low -=1
           
    # Final safety check for valid range
    if low >= high:
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
    filtered_mat = np.clip(filtered_mat, 0, 255).astype(np.uint8)

    return filtered_mat

# --- New function for aggregate statistics ---
def get_aggregate_filter_matrix(image_paths):
    print(f"Analyzing {len(image_paths)} images for aggregate statistics...")
    if not image_paths:
        print("Warning: No image paths provided for aggregate analysis.")
        # Return a default identity filter or raise an error
        return np.array([
            1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
        ])

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
        # Resize for consistent analysis speed & weighting (like original function)
        mat_resized = cv2.resize(mat_rgb, (256, 256))

        # Accumulate average values
        avg_bgr = cv2.mean(mat_resized)[:3] # Note: cv2.mean returns B,G,R order
        sum_avg_b += avg_bgr[0]
        sum_avg_g += avg_bgr[1]
        sum_avg_r += avg_bgr[2]


        # Accumulate histograms
        hist_r = cv2.calcHist([mat_resized], [0], None, [256], [0,256])
        hist_g = cv2.calcHist([mat_resized], [1], None, [256], [0,256])
        hist_b = cv2.calcHist([mat_resized], [2], None, [256], [0,256])
        sum_hist_r += hist_r
        sum_hist_g += hist_g
        sum_hist_b += hist_b

        # Accumulate total pixels (from resized)
        total_pixels += mat_resized.shape[0] * mat_resized.shape[1]
        valid_image_count += 1

    print("\nAggregate analysis complete.")

    if valid_image_count == 0:
        print("Error: No valid images found in the directory.")
        # Return a default identity filter
        return np.array([
            1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
        ])

    # Calculate overall average RGB
    # OpenCV mean is BGR, but we accumulated with R, G, B naming. Let's recalculate average correctly from sums.
    overall_avg_r = sum_avg_r / valid_image_count
    overall_avg_g = sum_avg_g / valid_image_count
    overall_avg_b = sum_avg_b / valid_image_count
    # avg_mat represents the average RGB pixel for hue shift calculation
    avg_mat = np.array([[[overall_avg_r, overall_avg_g, overall_avg_b]]], dtype=np.float32)


    # Find hue shift so that average red reaches MIN_AVG_RED
    # The hue shift needs to work on the *average* color vector, not the whole image average value
    # Let's calculate the hue shift based on the average pixel color vector directly
    hue_shift = 0
    current_avg_pixel = avg_mat.copy() # Start with the calculated average RGB
    # We need to simulate the *effect* of the hue shift on the R channel contribution.
    # The original logic `np.sum(shifted)` on a single pixel doesn't make sense.
    # Let's reinterpret: shift the *average pixel* and check its new red component.
    new_avg_r_component = current_avg_pixel[0,0,0]

    while(new_avg_r_component < MIN_AVG_RED):
        hue_shift += 1
        if hue_shift > MAX_HUE_SHIFT:
            # Prevent excessive shift, cap effect at MIN_AVG_RED equivalent shift
            print(f"Warning: Max hue shift ({MAX_HUE_SHIFT}) reached. Clamping red boost.")
            # Find the shift that *would* get it there if possible (approximate)
            # This part is tricky without re-applying to an image. Let's cap the shift value.
            hue_shift = MAX_HUE_SHIFT
            break # Exit loop

        shifted_avg_pixel = hue_shift_red(current_avg_pixel, hue_shift)
        new_avg_r_component = shifted_avg_pixel[0,0,0] # Check the red component of the shifted average pixel

    print(f"Calculated aggregate hue shift: {hue_shift}")

    # Note: The original logic applied the hue shift *back* to the image's red channel
    # before histogram analysis. This is complex with aggregate stats without processing
    # all pixels again. We will calculate the normalization based on the *original*
    # aggregate histograms but use the calculated `hue_shift` for the filter matrix gains.
    # This is a simplification but necessary for performance with directories.

    # Use aggregate histograms for normalization interval calculation
    normalize_mat = np.zeros((256, 3))
    # Use the total pixel count from all *resized* images for the threshold
    threshold_level = total_pixels / THRESHOLD_RATIO
    print(f"Aggregate pixel count: {total_pixels}, Threshold level: {threshold_level}")

    for x in range(256):
        if sum_hist_r[x] < threshold_level:
            normalize_mat[x][0] = x
        if sum_hist_g[x] < threshold_level:
            normalize_mat[x][1] = x
        if sum_hist_b[x] < threshold_level:
            normalize_mat[x][2] = x

    # Ensure 255 is always included
    normalize_mat[255][0] = 255
    normalize_mat[255][1] = 255
    normalize_mat[255][2] = 255

    adjust_r_low, adjust_r_high = normalizing_interval(normalize_mat[..., 0])
    adjust_g_low, adjust_g_high = normalizing_interval(normalize_mat[..., 1])
    adjust_b_low, adjust_b_high = normalizing_interval(normalize_mat[..., 2])

    print(f"Normalization R: ({adjust_r_low}, {adjust_r_high})")
    print(f"Normalization G: ({adjust_g_low}, {adjust_g_high})")
    print(f"Normalization B: ({adjust_b_low}, {adjust_b_high})")
    
    # Ensure intervals are valid
    adjust_r_high = max(adjust_r_high, adjust_r_low + 1)
    adjust_g_high = max(adjust_g_high, adjust_g_low + 1)
    adjust_b_high = max(adjust_b_high, adjust_b_low + 1)


    # Calculate filter matrix components using the aggregate hue_shift and normalization
    # Simulate hue shift on a white pixel [1,1,1] to get channel contributions
    # Use float32 for accuracy in hue shift calculation
    shifted = hue_shift_red(np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32), hue_shift)
    shifted_r, shifted_g, shifted_b = shifted[0][0]

    # Calculate gains and offsets
    red_gain_norm = 255.0 / (adjust_r_high - adjust_r_low)
    green_gain_norm = 255.0 / (adjust_g_high - adjust_g_low)
    blue_gain_norm = 255.0 / (adjust_b_high - adjust_b_low)

    # Original filter structure seemed to mix normalization gain and hue shift effects.
    # Let's try to map it:
    # R' = R*Kr*Hr + G*Kr*Hg + B*Kr*Hb + OR*255
    # G' = G*Kg     + OG*255
    # B' = B*Kb     + OB*255
    # Where K are normalization gains, H are hue shift contributions, O are offsets.

    # Calculate offsets based on normalization low points
    redOffset = (-adjust_r_low / 255.0) * red_gain_norm
    greenOffset = (-adjust_g_low / 255.0) * green_gain_norm
    blueOffset = (-adjust_b_low / 255.0) * blue_gain_norm

    # Combine hue shift factors with red normalization gain
    adjust_red = shifted_r * red_gain_norm
    adjust_red_green = shifted_g * red_gain_norm
    adjust_red_blue = shifted_b * red_gain_norm * BLUE_MAGIC_VALUE # Apply magic value here

    # Construct the filter matrix (simplified interpretation based on apply_filter structure)
    # Need to be careful with indices in apply_filter
    filter_matrix = np.array([
        # Row 1 (Output R = Inp R*c0 + Inp G*c1 + Inp B*c2 + c4*255)
        adjust_red, adjust_red_green, adjust_red_blue, 0, redOffset,
        # Row 2 (Output G = Inp R*c5 + Inp G*c6 + Inp B*c7 + c9*255) - Assuming G' depends only on G
        0, green_gain_norm, 0, 0, greenOffset,
         # Row 3 (Output B = Inp R*c10 + Inp G*c11 + Inp B*c12 + c14*255) - Assuming B' depends only on B
        0, 0, blue_gain_norm, 0, blueOffset,
         # Row 4 (Alpha/Unused)
        0, 0, 0, 1, 0,
    ], dtype=np.float32) # Use float for filter

    print("Calculated Aggregate Filter Matrix:\n", filter_matrix[:15].reshape(3,5)) # Print relevant parts

    return filter_matrix


# --- New function to process a directory ---
def correct_directory(input_dir, output_dir, output_prefix, yield_preview=False):
    # Find all image files in the input directory
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}"))) # Include uppercase extensions

    if not image_paths:
        yield "Error: No image files found in the selected directory.", 0, 0, None
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Calculate the single aggregate filter matrix
    yield "Status: Analyzing directory for aggregate filter...", 0, len(image_paths), None
    try:
        aggregate_filter_matrix = get_aggregate_filter_matrix(image_paths)
    except Exception as e:
         yield f"Error during analysis: {e}", 0, len(image_paths), None
         return


    # 2. Apply the filter to each image
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
                print(f"Warning: Could not read image {img_path} during application phase. Skipping.")
                yield status + " Skipped (Read Error).", i + 1, total_images, None
                continue

            # Convert to RGB for filtering
            rgb_mat = cv2.cvtColor(original_mat, cv2.COLOR_BGR2RGB)

            # Apply the *same* aggregate filter to this image
            corrected_mat_rgb = apply_filter(rgb_mat, aggregate_filter_matrix)

            # Convert back to BGR for saving
            corrected_mat_bgr = cv2.cvtColor(corrected_mat_rgb, cv2.COLOR_RGB2BGR)

            # Save the corrected image
            cv2.imwrite(output_path, corrected_mat_bgr)

            # Generate preview if requested
            if yield_preview:
                preview = original_mat.copy()
                width = preview.shape[1] // 2
                # Ensure corrected_mat_bgr has same dimensions as original_mat for preview merge
                if corrected_mat_bgr.shape == original_mat.shape:
                     preview[::, width:] = corrected_mat_bgr[::, width:]
                else: # Handle potential resize mismatches if any step changed dims (shouldn't happen here)
                     print(f"Warning: Dimension mismatch for preview {base_name}")
                     preview = corrected_mat_bgr # Show only corrected if mismatch

                preview_resized = cv2.resize(preview, (480, 270)) # Smaller preview
                ret, png = cv2.imencode('.png', preview_resized)
                if ret:
                    preview_bytes = png.tobytes()


            yield status, i + 1, total_images, preview_bytes

        except Exception as e:
            error_msg = f"Error processing {base_name}: {e}"
            print(error_msg)
            yield status + f" Error ({e})", i + 1, total_images, None # Update status with error

    yield f"Status: Finished processing {total_images} images.", total_images, total_images, None


# --- Original single image correction (kept for potential use/comparison) ---
def get_filter_matrix(mat):

    mat_resized = cv2.resize(mat, (256, 256)) # Use resized for analysis

    # Get average values of RGB
    avg_mat_bgr = cv2.mean(mat_resized)[:3]
    # Create a 1x1 pixel image with the average BGR color, then convert to RGB for hue shift
    avg_pixel_bgr = np.array([[[avg_mat_bgr[0], avg_mat_bgr[1], avg_mat_bgr[2]]]], dtype=np.float32)
    avg_pixel_rgb = cv2.cvtColor(avg_pixel_bgr, cv2.COLOR_BGR2RGB)


    # Find hue shift so that average red reaches MIN_AVG_RED
    hue_shift = 0
    current_avg_pixel = avg_pixel_rgb.copy()
    new_avg_r_component = current_avg_pixel[0,0,0]


    while(new_avg_r_component < MIN_AVG_RED):
        hue_shift += 1
        if hue_shift > MAX_HUE_SHIFT:
            hue_shift = MAX_HUE_SHIFT
            break

        shifted_avg_pixel = hue_shift_red(current_avg_pixel, hue_shift)
        new_avg_r_component = shifted_avg_pixel[0,0,0]

    # Apply hue shift to whole image's red channel contribution (Original approach)
    shifted_mat = hue_shift_red(mat_resized.astype(np.float32), hue_shift)
    new_r_channel = np.sum(shifted_mat, axis=2)
    new_r_channel = np.clip(new_r_channel, 0, 255)
    # Create a temporary mat with the modified red channel for histogram analysis
    mat_for_hist = mat_resized.copy()
    mat_for_hist[..., 0] = new_r_channel.astype(np.uint8)


    # Get histogram of all channels (using the temp mat with shifted red influence)
    hist_r = cv2.calcHist([mat_for_hist], [0], None, [256], [0,256])
    hist_g = cv2.calcHist([mat_for_hist], [1], None, [256], [0,256]) # Use original G,B channels
    hist_b = cv2.calcHist([mat_for_hist], [2], None, [256], [0,256])

    normalize_mat = np.zeros((256, 3))
    threshold_level = (mat_resized.shape[0]*mat_resized.shape[1])/THRESHOLD_RATIO # Use resized dimensions
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

    # Ensure intervals are valid
    adjust_r_high = max(adjust_r_high, adjust_r_low + 1)
    adjust_g_high = max(adjust_g_high, adjust_g_low + 1)
    adjust_b_high = max(adjust_b_high, adjust_b_low + 1)


    shifted = hue_shift_red(np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32), hue_shift)
    shifted_r, shifted_g, shifted_b = shifted[0][0]

    red_gain_norm = 255.0 / (adjust_r_high - adjust_r_low)
    green_gain_norm = 255.0 / (adjust_g_high - adjust_g_low)
    blue_gain_norm = 255.0 / (adjust_b_high - adjust_b_low)

    redOffset = (-adjust_r_low / 255.0) * red_gain_norm
    greenOffset = (-adjust_g_low / 255.0) * green_gain_norm
    blueOffset = (-adjust_b_low / 255.0) * blue_gain_norm

    adjust_red = shifted_r * red_gain_norm
    adjust_red_green = shifted_g * red_gain_norm
    adjust_red_blue = shifted_b * red_gain_norm * BLUE_MAGIC_VALUE

    # Construct the filter matrix
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