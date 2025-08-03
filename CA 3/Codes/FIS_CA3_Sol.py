import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import cv2
import glob
from scipy.signal import savgol_filter

##########################################################################################
################################### PARAMETER SECTION ####################################

anomaly_factor = 20  # Adjust this threshold as needed. typical is 20 works well

prestimulus = 0.5  # time before the event (adjust as needed)

time_resolution = 0.01  # Set time resolution in seconds (e.g., 50 ms)

interpolate_bads = False

preprocess_params = {'chs_to_remove':['Corr', 'Left Ear'], 
                    'filter_l':0.1, 
                    'filter_h':100, 
                    'notch_filter':60,  # find the optimum frequency based on PSD
                    'ref_channels':None,  # 
                    'resample':200} 

# Set paths for your data
base_data_path = r'C:\Users\Mohammadreza\Desktop\University\TA\FIS\Home Works\CA 3\Data'  # Current directory
output_base_path = r'C:\Users\Mohammadreza\Desktop\University\TA\FIS\Home Works\CA 3\Results'  # Directory to save results
##########################################################################################

# Define event IDs
start_reference_event_id = 230  # Start event code
end_event_id = 245  # End event code

# Function to preprocess EEG data


def preprocess_eeg(raw, preprocess_params):
    """
    Preprocess the raw EEG data:
    1. Remove 'Left Ear' and 'corr' channels
    2. Re-reference to the average reference
    3. Apply band-pass filter (0.5 Hz - 100 Hz)
    4. Apply notch filter at 60 Hz
    5. Downsample to 200 Hz
    """
    # 1. Remove 'Left Ear' and 'corr' channels if present
    channels_to_remove = preprocess_params['chs_to_remove']
    if preprocess_params['chs_to_remove'] is not None:
        print('Removing channels: ', preprocess_params['chs_to_remove'])
        raw.drop_channels([ch for ch in channels_to_remove if ch in raw.ch_names])

    # 2. Re-reference to the average reference
    if preprocess_params['ref_channels'] is not None:
        print('Re-referencing to channels: ', preprocess_params['ref_channels'])
        raw.set_eeg_reference(ref_channels=preprocess_params['ref_channels'])

    # 3. Apply a band-pass filter (0.5 Hz to 100 Hz)
    if (preprocess_params['filter_l'] is not None) or (preprocess_params['filter_h'] is not None):
        print('Filtering the signals in range: ', [preprocess_params['filter_l'], preprocess_params['filter_h']])
        raw.filter(l_freq=preprocess_params['filter_l'], h_freq=preprocess_params['filter_h'])

    # 4. Apply a notch filter at 60 Hz to remove power line noise
    if preprocess_params['notch_filter'] is not None:
        print('Notch filtering EEG in: ', preprocess_params['notch_filter'])
        raw.notch_filter(freqs=preprocess_params['notch_filter'])

    # 5. Downsample to 200 Hz
    if preprocess_params['resample'] is not None:
        print('Resampling to: ', preprocess_params['resample'])
        raw.resample(sfreq=preprocess_params['resample'])

    return raw


def create_video_from_frames(image_folder, video_name, fps=10):
    """Create a video from a series of images."""
    images = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    if not images:
        print(f"No images found in {image_folder}")
        return
    
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    
    # Use the 'mp4v' codec for MP4 videos
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Change the video file extension to .mp4
    if not video_name.endswith('.mp4'):
        video_name = os.path.splitext(video_name)[0] + '.mp4'
    
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
    
    for image in images:
        video.write(cv2.imread(image))
    
    cv2.destroyAllWindows()
    video.release()
    print(f"Video created: {video_name}")


# Function to handle subject processing and saving results
def calculate_ERP_and_TopoMap():
    # Load the EEG data
    eeg_vhdr_file_path = os.path.join(base_data_path, '901Goalie_Spring2024.vhdr')
    csv_trial_info_file_path = os.path.join(base_data_path, 'TrialInfo.csv')
    
    # Load the data
    raw = mne.io.read_raw_brainvision(eeg_vhdr_file_path, preload=True)

    if preprocess_params:
        raw = preprocess_eeg(raw, preprocess_params)
    
    # Load the standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    # Get the positions of the channels
    positions = montage.get_positions()['ch_pos']
    # Define the shift
    shift = np.array([0.0, 0.01, 0.03])
    # Apply the shift to all channel positions
    shifted_positions = {ch: pos + shift for ch, pos in positions.items() if ch in raw.ch_names}
    # Create a new montage with shifted positions
    custom_montage = mne.channels.make_dig_montage(ch_pos=shifted_positions, coord_frame='head')
    # Apply the custom montage to your data
    raw.set_montage(custom_montage)

    ############################################# Interpolate Bad Channels ############################################# 
    if interpolate_bads:
        channels_to_remove = ['FC1']
        raw.info['bads'].extend(channels_to_remove)
        
        raw.interpolate_bads(reset_bads=True)  # Interpolate and mark as good
    ##################################################################################################################### 
    
    # Load trial information
    trial_info = pd.read_csv(csv_trial_info_file_path)
    
    # Extract trial types and good trial information
    labels = trial_info['TrialType'].values
    good_trials = trial_info['GoodTrial'].values
    
    # Set parameters for sliding window
    sampling_rate = raw.info['sfreq']
    
    # Find events from the annotations
    events, event_id = mne.events_from_annotations(raw)

    print(f"Event IDs: {event_id}")
    
    # Find start and end events
    start_reference_events = events[events[:, 2] == start_reference_event_id]
    end_events = events[events[:, 2] == end_event_id]
    
    print(f"Number of start events: {len(start_reference_events)}")
    print(f"Number of end events: {len(end_events)}")

    # Step 1: Get the maximum trial length
    max_trial_length = 0
    trial_lengths = []
    change_intervals = []  # Store change intervals

    for i, start_reference in enumerate(start_reference_events):
        if i >= len(good_trials):
            print(f"Warning: Index {i} exceeds length of good_trials ({len(good_trials)})")
            break
            
        # Skip this trial if it's not a "good" trial
        if good_trials[i] == 0:
            continue

        start_time = start_reference[0] - int(prestimulus * sampling_rate)
        end = end_events[end_events[:, 0] > start_time]

        if len(end) == 0:
            continue
        end_time = end[0, 0]

        trial_length = end_time - start_time
        trial_lengths.append(trial_length)
        max_trial_length = max(max_trial_length, trial_length)

        # Extract event data to compute change interval
        event_data = raw.get_data(start=start_time, stop=end_time)

        # Compute change interval (max amplitude - min amplitude)
        change_interval = np.max(event_data, axis=1) - np.min(event_data, axis=1)
        change_intervals.append(change_interval)

    # Step 2: Count presence at each time point across trials and calculate acceptable change intervals
    presence_counts = np.zeros(max_trial_length, dtype=int)

    for i, start_reference in enumerate(start_reference_events):
        if i >= len(good_trials):
            break
            
        # Skip this trial if it's not a "good" trial
        if good_trials[i] == 0:
            continue

        start_time = start_reference[0] - int(prestimulus * sampling_rate)
        end = end_events[end_events[:, 0] > start_time]
        if len(end) == 0:
            continue
        end_time = end[0, 0]

        # Mark the time points that are present for this trial
        trial_start_idx = 0
        trial_end_idx = end_time - start_time
        presence_counts[trial_start_idx:trial_end_idx] += 1

    # Convert change intervals to NumPy array
    change_intervals = np.array(change_intervals)

    # Compute median and IQR
    median_change = np.median(change_intervals, axis=0)
    iqr_change = np.percentile(change_intervals, 75, axis=0) - np.percentile(change_intervals, 25, axis=0)

    # Define anomaly bounds
    lower_bound = median_change - anomaly_factor * iqr_change
    upper_bound = median_change + anomaly_factor * iqr_change

    # Step 3: Determine the cut-off time point
    threshold = 0.3 * len(trial_lengths)  # 30% of trials must remain
    cutoff_idx = max_trial_length

    for t, count in enumerate(presence_counts):
        if count < threshold:
            cutoff_idx = t
            break

    # Step 4: Trim the trials based on the cut-off point
    # Initialize containers for ERP calculations
    new_max_trial_length = 0
    left_event_data = []
    right_event_data = []

    for i, start_reference in enumerate(start_reference_events):
        if i >= len(good_trials) or i >= len(labels):
            break
            
        # Skip this trial if it's not a "good" trial
        if good_trials[i] == 0:
            continue

        start_time = start_reference[0] - int(prestimulus * sampling_rate)
        end = end_events[end_events[:, 0] > start_time]
        if len(end) == 0:
            continue
        end_time = end[0, 0]

        # Trim the trial if it exceeds the cut-off time
        if end_time - start_time > cutoff_idx:
            end_time = start_time + cutoff_idx

        # Get the event data for this trial
        event_data = raw.get_data(start=start_time, stop=end_time)

        # Compute change interval (max amplitude - min amplitude)
        channel_change_intervals = np.max(event_data, axis=1) - np.min(event_data, axis=1)

        # Check if the change interval is within the acceptable range
        for ch_idx, channel_change_interval in enumerate(channel_change_intervals):
            if channel_change_interval < lower_bound[ch_idx] or channel_change_interval > upper_bound[ch_idx]:
                print(f"Anomalous trial detected with unusual change interval! Start time: {start_time/raw.info['sfreq']}, End time: {end_time/raw.info['sfreq']}, Channel {raw.info['ch_names'][ch_idx]}, Change interval: {channel_change_interval}. REPLACING ALL THIS CHANNEL WITH NAN VALUES.")
                # Replace the entire row of this channel with NaNs
                event_data[ch_idx, :] = np.nan

        # Calculate the length of the current trial
        trial_length = end_time - start_time

        # Update the max trial length if this trial is longer
        if trial_length > new_max_trial_length:
            new_max_trial_length = trial_length

        # Apply baseline correction - use prestimulus period
        baseline_window = int(prestimulus * sampling_rate)
        if event_data.shape[1] > baseline_window:
            baseline_mean = np.mean(event_data[:, :baseline_window], axis=1, keepdims=True)
            event_data -= baseline_mean  # Remove baseline offset
        
        # Append the corrected data to the respective list based on label
        label = labels[i]

        if label == 'left':
            left_event_data.append(event_data)
        elif label == 'right':
            right_event_data.append(event_data)

    print('Number of trials remained:', len(left_event_data + right_event_data), ' - Left trials:', len(left_event_data), ', Right trials:', len(right_event_data))

    # Pad segments with NaNs
    def pad_segments(segments, max_length):
        padded_segments = []
        for segment in segments:
            padding = np.full((segment.shape[0], max_length - segment.shape[1]), np.nan)  # Fill with NaNs
            padded_segments.append(np.hstack((segment, padding)))
        return padded_segments
    
    left_event_data = pad_segments(left_event_data, new_max_trial_length)
    right_event_data = pad_segments(right_event_data, new_max_trial_length)

    mean_channels_left = np.nanmean(left_event_data, axis=0)
    mean_channels_right = np.nanmean(right_event_data, axis=0)
  
    erp_left = np.nanmean(mean_channels_left, axis=0)
    erp_right = np.nanmean(mean_channels_right, axis=0)

    # Calculate standard deviation across trials and channels, ignoring NaNs
    std_left = np.nanstd(mean_channels_left, axis=0)
    std_right = np.nanstd(mean_channels_right, axis=0)

    # Calculate SEM (Standard Error of the Mean)
    n_left = len(left_event_data)  # Number of trials for 'Left' condition
    n_right = len(right_event_data)  # Number of trials for 'Right' condition
    sem_left = std_left / np.sqrt(n_left)
    sem_right = std_right / np.sqrt(n_right)

    # Compute the min and max of both Left and Right ERPs to determine the vlim range for topomap
    min_left_erp = np.nanmin(erp_left)
    max_left_erp = np.nanmax(erp_left)
    min_right_erp = np.nanmin(erp_right)
    max_right_erp = np.nanmax(erp_right)

    # Combine the min and max values from both ERPs to determine the overall vlim range
    vlim_min = min(min_left_erp, min_right_erp)
    vlim_max = max(max_left_erp, max_right_erp)

    # Set vlim as a tuple (min, max) for the topomap
    vlim = (vlim_min, vlim_max)

    # Plot ERP averaged across channels and save with prestimulus period
    time_axis = np.linspace(-prestimulus, 
                            (new_max_trial_length - int(prestimulus * sampling_rate)) / sampling_rate, 
                            new_max_trial_length)
    
    # Create output directory
    subject = '901Goalie_Spring2024'
    mode = 'eeg_analysis'
    output_folder = os.path.join(output_base_path, subject)
    os.makedirs(output_folder, exist_ok=True)

    # Time points to plot, based on the chosen time resolution
    time_points = np.arange(time_axis[0], time_axis[-1], time_resolution)

    # Loop through each time point
    for idx, time_point in enumerate(time_points):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot the topomap for 'Left' class at the current time point
        plot_topomap(mean_channels_left, raw.info, vlim, axes[0], "Left", time_point, prestimulus)
        
        # Plot the topomap for 'Right' class at the current time point
        plot_topomap(mean_channels_right, raw.info, vlim, axes[1], "Right", time_point, prestimulus)

        # Plot the ERP for 'Left' and 'Right' at the current time point
        axes[2].plot(time_axis, erp_left, label='Left', color='teal')
        axes[2].fill_between(time_axis, erp_left - sem_left, erp_left + sem_left, color='teal', alpha=0.2)
        axes[2].plot(time_axis, erp_right, label='Right', color='magenta')
        axes[2].fill_between(time_axis, erp_right - sem_right, erp_right + sem_right, color='magenta', alpha=0.2)
        axes[2].axvline(x=0, color='black', linestyle='--', alpha=0.1)  # Marking event 230 as 0
        axes[2].axvline(x=time_point, color='black', linestyle='--')  # Mark the current time point
        axes[2].set_title(f"ERP at {time_point:.3f}s")
        axes[2].legend(loc='best', fontsize=12)  # This ensures that the legend is drawn

        # Save the figure
        output_path = os.path.join(output_folder, f"{subject}_{mode}_Topomap_{idx+1:03d}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()  # Close the figure to free up memory
    
    # Create a video from the saved frames
    video_path = os.path.join(output_base_path, f"{subject}_{mode}_topomap_video.mp4")
    create_video_from_frames(output_folder, video_path)
    
    return output_folder, video_path

def plot_topomap(data, info, vlim, ax, title, time_point, prestimulus):
    """Plot the topomap for the given data at the specified time."""
    # Adjust time_point relative to the stimulus (subtract prestimulus time)
    adjusted_time_point = time_point + prestimulus  # Make sure time_point is positive or zero
    time_idx = int(adjusted_time_point * info['sfreq'])  # Convert adjusted time to sample index

    # Ensure time_idx is within bounds
    time_idx = max(0, min(time_idx, data.shape[1] - 1))  # Clip to valid index range
    
    topomap_data = data[:, time_idx] 

    # Plot topomap
    im, _ = mne.viz.plot_topomap(
        topomap_data, 
        names=info['ch_names'], 
        vlim=vlim, 
        pos=info, 
        axes=ax, 
        show=False
        )
    # Set the title for the axis
    ax.set_title(title) 

    # Add the colorbar (color scale) for the topomap
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label('Amplitude (µV)', rotation=270, labelpad=15)  # You can adjust the label based on your data's units

# Run the main function
if __name__ == "__main__":
    output_folder, video_path = calculate_ERP_and_TopoMap()
    print(f"Analysis completed. Results saved to {output_folder}")
    print(f"Video saved to {video_path}")