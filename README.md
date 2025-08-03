# Functional Brain Imaging Systems - Course Assignments

This repository contains the homework assignments for the Master's level course, **Functional Brain Imaging Systems**.

---

- **Instructor:** Dr. Ali Khadem
- **University:** K. N. Toosi University of Technology (KNTU)
- **Department:** Biomedical Engineering Group, Faculty of Electrical Engineering

These assignments were designed to provide students with hands-on experience in processing and analyzing electroencephalography (EEG) data using Python. They were developed by Mohammadreza Shahsavari during his role as a Teaching Assistant for the course.

---

##  Assignments Overview

The repository contains three major assignments that build upon each other, guiding students from fundamental preprocessing to advanced data analysis techniques.

### **Assignment 3: EEG Preprocessing and ERP Analysis**
* **Objective:** To introduce the fundamental pipeline for cleaning, preparing, and analyzing EEG data to find event-related responses.
* **Tasks:**
    * Working with EEG data from a participant trying to predict the actions of others (data collected at [MVP Lab](https://mvaziri.github.io/Homepage/mvpLab.html) and used with permission).
    * Performing essential preprocessing steps including filtering, epoching, and baseline correction.
    * Identifying and interpolating bad channels to handle artifacts.
    * Calculating and plotting Event-Related Potentials (ERPs) for different experimental conditions.
    * Visualizing brain activity by creating animated topomaps of scalp voltage.
* **Key Libraries:** MNE-Python, NumPy, Matplotlib.

### **Assignment 5: Machine Learning for Parkinson's Disease Classification**
* **Objective:** To apply machine learning techniques to classify participants as Parkinson's Disease (PD) patients or Healthy Controls (HC) based on their resting-state EEG data.
* **Tasks:**
    * Extracting Power Spectral Density (PSD) features from different frequency bands of the EEG signals.
    * Training and evaluating various machine learning classifiers, including SVM, K-Nearest Neighbors, and Random Forest.
    * Implementing a cross-validation strategy to ensure robust model evaluation.
    * Reporting standard performance metrics such as Accuracy, Precision, Recall, and F1-score.
* **Key Libraries:** MNE-Python, NumPy, Scikit-learn, Matplotlib.

### **Final Project: Multivariate Pattern Analysis (MVPA)**
* **Objective:** To investigate the spatiotemporal dynamics of the brain during an action prediction task using advanced decoding methods.
* **Tasks:**
    * Applying a time-resolved decoding approach (MVPA) to distinguish between experimental conditions ('Left Button Pressed' vs. 'Right Button Pressed') from EEG data.
    * Training a classifier (e.g., SVM) at each individual time point of the EEG epochs.
    * Plotting the time course of decoding accuracy to identify when the brain discriminates between the two conditions.
    * Performing cluster-based permutation tests to find significant spatiotemporal patterns of brain activity.
* **Key Libraries:** MNE-Python, Scikit-learn, NumPy, Matplotlib.

---

## 📂 Repository Structure

The repository is organized into directories for each assignment. Inside each directory (`CA 3`, `CA 5`, `Final Project`), you will find:
* A **PDF file** with the detailed problem statement and instructions.
* A `Codes` folder containing the Python solution scripts.
* `Data` and `Results` folders with sample data and visualizations (e.g., plots, videos).

---

## 🚀 Getting Started

To run the code in this repository, you will need a Python environment with the following key libraries installed:

```bash
pip install mne numpy scipy pandas matplotlib scikit-learn
