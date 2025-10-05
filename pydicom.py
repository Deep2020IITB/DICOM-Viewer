# -*- coding: utf-8 -*-
"""app.py

Streamlit-based DICOM Viewer
"""
!pip install streamlit pydicom matplotlib numpy nibabel -q

import os
import zipfile
import numpy as np
import pydicom
from collections import defaultdict
import matplotlib.pyplot as plt
import streamlit as st
import tempfile
import shutil

# Streamlit page configuration (must be the first Streamlit command)
if not hasattr(st.session_state, 'page_config_set'):
    st.set_page_config(page_title="DICOM Viewer", layout="wide")
    st.session_state.page_config_set = True

# Initialize session state for persistent variables
if 'series_groups' not in st.session_state:
    st.session_state.series_groups = defaultdict(list)
if 'series_custom_names' not in st.session_state:
    st.session_state.series_custom_names = {}
if 'volume' not in st.session_state:
    st.session_state.volume = None
if 'selected_series_desc' not in st.session_state:
    st.session_state.selected_series_desc = ''
if 'selected_series_uid' not in st.session_state:
    st.session_state.selected_series_uid = None
if 'extract_dir' not in st.session_state:
    st.session_state.extract_dir = tempfile.mkdtemp()

# Function to extract ZIP and list series
def extract_and_list_series(uploaded_file):
    series_groups = defaultdict(list)
    series_custom_names = {}

    if not uploaded_file:
        st.error("Please upload a valid ZIP file.")
        return series_groups, series_custom_names

    # Save uploaded ZIP to temporary directory
    zip_path = os.path.join(st.session_state.extract_dir, uploaded_file.name)
    with open(zip_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Extract ZIP
    extract_path = os.path.join(st.session_state.extract_dir, uploaded_file.name.replace('.zip', ''))
    try:
        os.makedirs(st.session_state.extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        st.success(f"Extracted ZIP to {extract_path}")
    except Exception as e:
        st.error(f"Error extracting {uploaded_file.name}: {e}")
        return series_groups, series_custom_names

    # Find any folder containing DICOM files or use the root if DICOM files are there
    study_folder = None
    for subdir in os.listdir(extract_path):
        subdir_path = os.path.join(extract_path, subdir)
        if os.path.isdir(subdir_path):
            # Check if the subfolder contains any .dcm files
            for root, _, files in os.walk(subdir_path):
                if any(file.endswith(".dcm") for file in files):
                    study_folder = subdir_path
                    break
        if study_folder:
            break
    
    # If no subfolder with DICOM files is found, check the root of the extracted path
    if not study_folder:
        for file in os.listdir(extract_path):
            if file.endswith(".dcm"):
                study_folder = extract_path
                break
    
    if not study_folder:
        st.error(f"No folder containing DICOM (.dcm) files found in {extract_path}.")
        return series_groups, series_custom_names

    # Verify pydicom.dcmread is available
    if not hasattr(pydicom, 'dcmread'):
        st.error("Error: pydicom.dcmread is not available. Ensure pydicom>=1.0.0 is installed correctly.")
        return series_groups, series_custom_names

    # Group DICOM files by SeriesInstanceUID
    for subdir, _, files in os.walk(study_folder):
        for file in files:
            if file.endswith(".dcm"):
                file_path = os.path.join(subdir, file)
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    series_uid = ds.SeriesInstanceUID if 'SeriesInstanceUID' in ds else subdir
                    series_groups[series_uid].append((file_path, ds))
                except Exception as e:
                    st.error(f"Error reading {file_path}: {e}")

    # Sort slices
    for uid, files in series_groups.items():
        files.sort(key=lambda x: (x[1].get('InstanceNumber', 0), x[1].get('SliceLocation', 0)))

    # Populate series options
    series_options = []
    for uid, files in series_groups.items():
        if files:
            ds = files[0][1]
            series_desc = ds.get('SeriesDescription', os.path.basename(os.path.dirname(files[0][0])))
            series_options.append((f"{series_desc} ({len(files)} slices)", uid))
            series_custom_names[uid] = series_desc

    if series_options:
        st.write("\n**Available Series:**")
        for desc, uid in series_options:
            ds = series_groups[uid][0][1]
            st.write(f"**Series:** {desc} (UID: {uid})")
            st.write(f"  **Modality:** {ds.get('Modality', 'N/A')}")
            st.write(f"  **Slices:** {len(series_groups[uid])}")
            st.write(f"  **Patient:** {ds.get('PatientName', 'N/A')} | **Date:** {ds.get('StudyDate', 'N/A')}")
            st.write("---")
    else:
        st.error(f"No valid DICOM series found in {study_folder}.")

    return series_groups, series_custom_names

# Function to process selected series
def process_series(series_uid, series_groups, series_custom_names):
    if not series_uid:
        st.error("Please select a valid series.")
        return None, None

    selected_series_desc = series_custom_names.get(series_uid,
        series_groups[series_uid][0][1].get('SeriesDescription',
        os.path.basename(os.path.dirname(series_groups[series_uid][0][0]))))

    # Verify pydicom.dcmread is available
    if not hasattr(pydicom, 'dcmread'):
        st.error("Error: pydicom.dcmread is not available. Ensure pydicom>=1.0.0 is installed correctly.")
        return None, None

    # Load pixel data
    volume = []
    for file_path, ds in series_groups[series_uid]:
        try:
            ds = pydicom.dcmread(file_path)
            pixel_data = ds.pixel_array
            # Normalize for MRI
            pixel_data = np.clip(pixel_data, 0, np.percentile(pixel_data, 99))
            pixel_data = (pixel_data - pixel_data.min()) / (pixel_data.max() - pixel_data.min() + 1e-6)
            volume.append(pixel_data)
        except Exception as e:
            st.error(f"Error loading pixels from {file_path}: {e}")

    if volume:
        volume = np.stack(volume)
        st.success(f"Loaded volume shape: {volume.shape}")
        return volume, selected_series_desc
    else:
        st.error("No valid pixel data loaded from selected series.")
        return None, None

# Main Streamlit app
st.title("DICOM Viewer")

# File uploader for ZIP
uploaded_file = st.file_uploader("Upload a ZIP file containing DICOM data", type="zip")

# Button to list series
if st.button("List Series"):
    if uploaded_file:
        # Clear previous extraction directory
        if os.path.exists(st.session_state.extract_dir):
            shutil.rmtree(st.session_state.extract_dir)
        st.session_state.extract_dir = tempfile.mkdtemp()
        st.session_state.series_groups, st.session_state.series_custom_names = extract_and_list_series(uploaded_file)
    else:
        st.error("Please upload a ZIP file first.")

# Series selection
series_options = ['Select a series']
series_uids = []
if st.session_state.series_groups:
    for uid, files in st.session_state.series_groups.items():
        if files:
            ds = files[0][1]
            series_desc = st.session_state.series_custom_names.get(uid,
                ds.get('SeriesDescription', os.path.basename(os.path.dirname(files[0][0]))))
            series_options.append(f"{series_desc} ({len(files)} slices)")
            series_uids.append(uid)

selected_series = st.selectbox("Select a Series", series_options)

# Series name input
if selected_series != 'Select a series':
    selected_idx = series_options.index(selected_series) - 1
    if selected_idx >= 0:
        selected_uid = series_uids[selected_idx]
        st.session_state.selected_series_uid = selected_uid
        default_name = st.session_state.series_custom_names.get(selected_uid,
            st.session_state.series_groups[selected_uid][0][1].get('SeriesDescription',
            os.path.basename(os.path.dirname(st.session_state.series_groups[selected_uid][0][0]))))
        series_name = st.text_input("Custom Series Name", value=default_name, key=f"series_name_{selected_uid}")
        if series_name != default_name:
            st.session_state.series_custom_names[selected_uid] = series_name
            # Update series dropdown
            series_options[series_options.index(selected_series)] = f"{series_name} ({len(st.session_state.series_groups[selected_uid])} slices)"
            st.session_state.series_custom_names[selected_uid] = series_name

# Process series button
if st.button("Process Series") and selected_series != 'Select a series':
    selected_idx = series_options.index(selected_series) - 1
    if selected_idx >= 0:
        selected_uid = series_uids[selected_idx]
        st.session_state.volume, st.session_state.selected_series_desc = process_series(
            selected_uid, st.session_state.series_groups, st.session_state.series_custom_names)

        if st.session_state.volume is not None:
            # Get DICOM metadata for display
            ds = st.session_state.series_groups[selected_uid][0][1]
            metadata = {
                'PatientName': str(ds.get('PatientName', 'N/A')),
                'Slices': len(st.session_state.series_groups[selected_uid]),
                'Modality': ds.get('Modality', 'N/A'),
                'StudyDate': ds.get('StudyDate', 'N/A'),
                'SliceThickness': ds.get('SliceThickness', 'N/A'),
                'SliceLocation': ds.get('SliceLocation', 'N/A'),
                'RepetitionTime': ds.get('RepetitionTime', 'N/A'),
                'EchoTime': ds.get('EchoTime', 'N/A'),
                'FlipAngle': ds.get('FlipAngle', 'N/A')
            }

            # Slice viewer
            slice_idx = st.slider("Select Slice", 0, len(st.session_state.volume) - 1, len(st.session_state.volume) // 2)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(st.session_state.volume[slice_idx], cmap='gray')
            ax.set_title(
                f"Slice {slice_idx} from {st.session_state.selected_series_desc}\n\n"
                f"Patient Name: {metadata['PatientName']}\n"
                f"Number of Slices: {metadata['Slices']}\n"
                f"Modality: {metadata['Modality']}\n"
                f"Date: {metadata['StudyDate']}\n"
                f"ST: {metadata['SliceThickness']} | SL: {metadata['SliceLocation']}\n"
                f"RT: {metadata['RepetitionTime']} | ET: {metadata['EchoTime']} | FS: {metadata['FlipAngle']}",
                fontsize=10
            )
            ax.axis('off')
            st.pyplot(fig)

            # Optional NIfTI export
            if st.button("Export as NIfTI"):
                try:
                    import nibabel as nib
                    nifti_path = os.path.join(st.session_state.extract_dir,
                        f"{st.session_state.selected_series_desc.replace(' ', '_')}_volume.nii.gz")
                    nifti_img = nib.Nifti1Image(st.session_state.volume, affine=np.eye(4))  # Simple affine
                    nib.save(nifti_img, nifti_path)
                    st.success(f"Saved NIfTI to {nifti_path}")
                    with open(nifti_path, 'rb') as f:
                        st.download_button("Download NIfTI", f, file_name=os.path.basename(nifti_path))
                except ImportError:
                    st.warning("NIfTI export requires nibabel. Install it with `pip install nibabel`.")
