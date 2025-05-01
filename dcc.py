# /dcc.py:
#--------------------------------------------------------------------------------
import PySimpleGUI as sg
import os
# Updated imports: added correct_directory
from correct import correct_image, analyze_video, process_video, correct_directory
import webbrowser
from logo.logo import LOGO
import threading # Added for smoother GUI updates during processing
import queue # Added for thread communication

IMAGE_TYPES = (".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff") # Added TIFF
VIDEO_TYPES = (".mp4", ".mkv", ".avi", ".mov")

sg.theme('DarkGrey6')
sg.set_options(font=("Arial", 13))
sg.set_global_icon(LOGO)

# --- GUI Layout Changes ---
left_column = [
    [
        sg.Text("Process Mode:", size=(15, 1)),
        sg.Radio("Files/Videos", "MODE", default=True, key="__MODE_FILES__", enable_events=True, size=(12,1)),
        sg.Radio("Directory", "MODE", key="__MODE_DIR__", enable_events=True, size=(12,1))
    ],
    # File Selection (visible by default)
    [ sg.Frame("File/Video Selection", [
        [sg.FilesBrowse(button_text="Select Photos/Videos", enable_events=True, key='__INPUT_FILES__', size=(20,1))],
        [sg.Listbox(values=[], enable_events=True, size=(50, 15), key="__INPUT_FILE_LIST__")]
      ], key="__FILE_FRAME__", visible=True)
    ],
    # Directory Selection (hidden by default)
    [ sg.Frame("Directory Selection", [
        [
            sg.Text("Input Directory", size=(15,1)),
            sg.InputText(size=(28, 1), enable_events=True, readonly=True, key="__INPUT_DIRECTORY__", background_color='white'),
            sg.FolderBrowse(button_text="Select Dir")
        ]
      ], key="__DIR_FRAME__", visible=False) # Start hidden
    ],
    # Output settings common to both modes
    [
        sg.Text("Output Folder", size=(15, 1)),
        sg.InputText(default_text=os.path.expanduser("~"), size=(28, 1), enable_events=True, readonly=True, background_color='white', key="__OUTPUT_FOLDER__"), # Default to home
        sg.FolderBrowse(button_text="Select Out")
    ],
    [
        sg.Text(text="Output File Prefix", size=(15, 1)),
        sg.InputText(default_text="corrected", size=(28, 1), key="__OUTPUT_PREFIX__")
    ],
    [
        sg.Button(button_text="Correct All", enable_events=True, pad=(10, 10), button_color='#cc4827', key="__CORRECT__"),
        sg.Button(button_text="Cancel", enable_events=True, pad=(5, 10), disabled=True, key="__CANCEL__"),
        sg.Button(button_text="Clear List/Dir", enable_events=True, pad=(5, 10), disabled=False, key="__CLEAR_LIST__")
    ],
    [
        sg.Text(text="", size=(30, 1), text_color='yellow', key="__STATUS__"),
        sg.ProgressBar(100, orientation='h', size=(20, 20), key='__PROGRESS__', bar_color=('#cc4827', '#404040'), visible=False)
    ]
]

info = [
    [sg.Text("DCC: Dive Color Corrector", font=('Arial', 30))],
    [
        sg.Text("By"),
        sg.Text("@harsha_bornfree", pad=(1, 0), enable_events=True, text_color='khaki', key='__TWITTER_LINK__'),
    ],
    [sg.Text("", pad=(10, 5))],
    [sg.Text("An easy and free tool to color correct your dive videos and photos.")],
    [sg.Text("Select 'Files/Videos' or 'Directory' mode.")],
    [sg.Text("Add items using the 'Select' buttons.")],
    [sg.Text("Choose an output folder and prefix, then click 'Correct All'.")],
    [sg.Text("", pad=(10, 5))],
    [sg.Text("> Uses single-image stats for files/videos.")],
    [sg.Text("> Uses aggregate directory stats for 'Directory' mode.")],
    [sg.Text("> No watermarks, time limits, or locked features.")],
    [sg.Text("", pad=(10, 5))],
    [sg.Text("If you found this tool useful, please consider donating:", text_color='khaki')],
    [sg.Button("Donate", enable_events=True, key="__DONATION_LINK__")]
]

viewer = [
    [
        sg.Frame("", layout=info, key="__INFO__", expand_x=True, expand_y=True),
        sg.Image(visible=False, key="__PREVIEW__")
    ]
]

layout = [
    [
        sg.Column(left_column),
        # sg.VSeparator(), # Optional separator
        sg.Column(viewer, expand_x=True, expand_y=True)
    ]
]

window = sg.Window("DCC: Dive Color Corrector", layout, finalize=True, resizable=True) # Added resizable and finalize

# --- Global state variables ---
file_generator = None
file_index = 0
analyze_video_generator = None
process_video_generator = None
correct_directory_generator = None # Added
processing_thread = None # Added
cancel_event = threading.Event() # Added
gui_queue = queue.Queue() # Added

# --- Helper Functions ---
def valid_file(path):
    if not os.path.isfile(path):
        return False
    extension = os.path.splitext(path)[1].lower()
    return extension in IMAGE_TYPES or extension in VIDEO_TYPES

def get_files(filepaths):
    input_filepaths = [f for f in filepaths if valid_file(f)]
    for f in input_filepaths:
        yield f

def update_ui_state(processing=False):
    """Enable/disable buttons based on processing state"""
    window["__CORRECT__"].update(disabled=processing)
    window["__CANCEL__"].update(disabled=not processing)
    window["__CLEAR_LIST__"].update(disabled=processing)
    window["__MODE_FILES__"].update(disabled=processing)
    window["__MODE_DIR__"].update(disabled=processing)
    window["__INPUT_FILES__"].update(disabled=processing)
    # Keep input directory browse enabled maybe? Or disable too. Let's disable.
    window["__INPUT_DIRECTORY__"].update(disabled=processing)
    window["__OUTPUT_FOLDER__"].update(disabled=processing)
    window["__OUTPUT_PREFIX__"].update(disabled=processing)

def run_correction_thread(mode, input_data, output_folder, output_prefix):
    """Target function for the processing thread"""
    global gui_queue, cancel_event

    cancel_event.clear() # Reset cancel flag

    try:
        if mode == "files":
            file_list = list(input_data) # Consume generator for thread
            total_files = len(file_list)
            for i, f in enumerate(file_list):
                if cancel_event.is_set():
                    gui_queue.put(("status", "Cancelled"))
                    break

                new_filename = f"{output_prefix}_{os.path.basename(f)}"
                output_filepath = os.path.join(output_folder, new_filename)
                extension = os.path.splitext(f)[1].lower()
                progress = int(100 * (i + 1) / total_files) if total_files > 0 else 0

                if extension in IMAGE_TYPES:
                    gui_queue.put(("status", f"Correcting image {i+1}/{total_files}..."))
                    gui_queue.put(("progress", progress))
                    preview = correct_image(f, output_filepath)
                    if preview:
                        gui_queue.put(("preview", preview))
                elif extension in VIDEO_TYPES:
                    # --- Video Analysis ---
                    gui_queue.put(("status", f"Analyzing video {i+1}/{total_files}..."))
                    gui_queue.put(("progress", progress)) # Progress for analysis start
                    vid_analyzer = analyze_video(f, output_filepath)
                    video_data = None
                    for item in vid_analyzer:
                        if cancel_event.is_set(): break
                        if isinstance(item, dict):
                            video_data = item
                            if "error" in video_data:
                                gui_queue.put(("status", f"Video analysis failed: {video_data['error']}"))
                            break
                        elif isinstance(item, int):
                             gui_queue.put(("status", f"Analyzing V:{i+1}/{total_files} (Frame {item})..."))
                        # Add short sleep to allow GUI updates? time.sleep(0.01) # Requires import time
                    if cancel_event.is_set() or not video_data or "error" in video_data:
                         if not cancel_event.is_set():
                             gui_queue.put(("status", "Video analysis failed or cancelled."))
                         continue # Skip processing if analysis failed/cancelled

                    # --- Video Processing ---
                    gui_queue.put(("status", f"Processing video {i+1}/{total_files}..."))
                    vid_processor = process_video(video_data, True)
                    for percent, preview in vid_processor:
                        if cancel_event.is_set(): break
                        if percent is not None:
                            gui_queue.put(("status", f"Processing V:{i+1}/{total_files} ({percent:.1f}%)"))
                            gui_queue.put(("progress", int(percent))) # Update overall progress during video
                        if preview:
                            gui_queue.put(("preview", preview))
                    if cancel_event.is_set(): break
                else:
                     gui_queue.put(("status", f"Skipping unknown file type: {os.path.basename(f)}"))

            if not cancel_event.is_set():
                gui_queue.put(("status", "All files/videos processed!"))
                gui_queue.put(("progress", 100))

        elif mode == "directory":
            dir_processor = correct_directory(input_data, output_folder, output_prefix, True)
            for status_msg, current, total, preview in dir_processor:
                if cancel_event.is_set():
                    gui_queue.put(("status", "Cancelled"))
                    break
                progress = int(100 * current / total) if total > 0 else 0
                gui_queue.put(("status", status_msg))
                gui_queue.put(("progress", progress))
                if preview:
                    gui_queue.put(("preview", preview))
            if not cancel_event.is_set():
                 # Final status might already be in status_msg from generator
                 gui_queue.put(("progress", 100))


    except Exception as e:
        import traceback
        print("Error in processing thread:", traceback.format_exc())
        gui_queue.put(("status", f"Thread Error: {e}"))
    finally:
        # Signal completion (even if cancelled or error)
        gui_queue.put(("processing_complete", None))


# --- Main Event Loop ---
if __name__ == "__main__":
    while True:
        event, values = window.read(timeout=100) # Add timeout for queue checking

        # --- Process GUI queue messages from thread ---
        try:
            while True:
                message_type, message_data = gui_queue.get_nowait()
                if message_type == "status":
                    window["__STATUS__"].update(message_data)
                elif message_type == "progress":
                    window["__PROGRESS__"].update(current_count=message_data, visible=True)
                elif message_type == "preview":
                    window["__PREVIEW__"].update(data=message_data, visible=True)
                    window["__INFO__"].update(visible=False)
                elif message_type == "processing_complete":
                    update_ui_state(processing=False)
                    window["__PROGRESS__"].update(visible=False)
                    # Keep preview visible until cleared or new process starts
                    processing_thread = None # Clear thread variable
                    # Status might be set already, or set a final one if needed
                    # window["__STATUS__"].update("Processing finished.")

        except queue.Empty:
            pass # No messages in queue

        # --- Handle GUI events ---
        if event == sg.WIN_CLOSED:
            if processing_thread and processing_thread.is_alive():
                cancel_event.set() # Signal thread to stop
                processing_thread.join(timeout=2) # Wait briefly for thread
            break

        if event == "__TWITTER_LINK__":
            webbrowser.open("https://twitter.com/harsha_bornfree")

        if event == "__DONATION_LINK__":
            webbrowser.open("https://buy.stripe.com/28obMb8Mx2EEbRK7ss")

        # --- Mode Switching ---
        if event == "__MODE_FILES__":
            window["__FILE_FRAME__"].update(visible=True)
            window["__DIR_FRAME__"].update(visible=False)
            window["__INPUT_DIRECTORY__"].update("") # Clear directory selection
        elif event == "__MODE_DIR__":
            window["__FILE_FRAME__"].update(visible=False)
            window["__DIR_FRAME__"].update(visible=True)
            window["__INPUT_FILE_LIST__"].update([]) # Clear file selection

        # --- Input Selection ---
        if event == "__INPUT_FILES__":
            # Only process if in Files mode
            if values["__MODE_FILES__"]:
                existing_filepaths = window["__INPUT_FILE_LIST__"].get_list_values()
                # Handle potential empty string from cancel
                new_files = values["__INPUT_FILES__"].split(";") if values["__INPUT_FILES__"] else []
                filepaths = existing_filepaths + new_files

                input_filepaths = [f for f in filepaths if valid_file(f)]
                window["__INPUT_FILE_LIST__"].update(list(set(input_filepaths))) # Use set to remove duplicates

                # Auto-set output folder based on first *newly added* valid file if list was empty
                if not existing_filepaths and input_filepaths:
                     try:
                        window["__OUTPUT_FOLDER__"].update(os.path.dirname(input_filepaths[0]))
                     except Exception: # Handle potential index error if filtering results in empty list
                         pass


        # Directory selection is handled directly by FolderBrowse updating __INPUT_DIRECTORY__

        # --- Output Folder Selection ---
        if event == "__OUTPUT_FOLDER__":
            # Value is updated automatically by FolderBrowse if it returns a value
            # No explicit update needed here unless you want validation
            pass

        # --- Correct Button ---
        if event == "__CORRECT__":
            output_folder = values["__OUTPUT_FOLDER__"]
            output_prefix = values["__OUTPUT_PREFIX__"]

            if not output_folder or not os.path.isdir(output_folder):
                sg.popup_error("Please select a valid Output Folder.")
                continue
            if not output_prefix:
                sg.popup_error("Please enter an Output File Prefix.")
                continue

            # Check which mode is active
            if values["__MODE_FILES__"]:
                filepaths = window["__INPUT_FILE_LIST__"].get_list_values()
                if not filepaths:
                    sg.popup_error("No files selected for processing.")
                    continue
                file_generator = get_files(filepaths) # Create generator
                update_ui_state(processing=True)
                window["__STATUS__"].update("Starting file/video processing...")
                window["__PROGRESS__"].update(0, visible=True)
                window["__PREVIEW__"].update(visible=True) # Show preview area
                window["__INFO__"].update(visible=False)
                # Start thread for file/video mode
                processing_thread = threading.Thread(
                    target=run_correction_thread,
                    args=("files", file_generator, output_folder, output_prefix),
                    daemon=True)
                processing_thread.start()

            elif values["__MODE_DIR__"]:
                input_directory = values["__INPUT_DIRECTORY__"]
                if not input_directory or not os.path.isdir(input_directory):
                     sg.popup_error("Please select a valid Input Directory.")
                     continue
                update_ui_state(processing=True)
                window["__STATUS__"].update("Starting directory processing...")
                window["__PROGRESS__"].update(0, visible=True)
                window["__PREVIEW__"].update(visible=True) # Show preview area
                window["__INFO__"].update(visible=False)
                 # Start thread for directory mode
                processing_thread = threading.Thread(
                    target=run_correction_thread,
                    args=("directory", input_directory, output_folder, output_prefix),
                    daemon=True)
                processing_thread.start()

        # --- Cancel Button ---
        if event == "__CANCEL__":
            if processing_thread and processing_thread.is_alive():
                window["__STATUS__"].update("Cancelling...")
                cancel_event.set() # Signal the thread to stop
                # The thread will put the final "Cancelled" status and "processing_complete" signal
            else:
                # If somehow cancel is pressed when not processing
                update_ui_state(processing=False)
                window["__STATUS__"].update("Idle.")
                window["__PROGRESS__"].update(visible=False)


        # --- Clear Button ---
        if event == "__CLEAR_LIST__":
            window["__INPUT_FILE_LIST__"].update(values=[])
            window["__INPUT_DIRECTORY__"].update("")
            window["__STATUS__"].update("")
            window["__PREVIEW__"].update(visible=False) # Hide preview
            window["__INFO__"].update(visible=True)   # Show info


    window.close()