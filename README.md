# Functional Brain Imaging Systems - Course Assignments

Welcome to the repository for the homework assignments of the Master's level course, "Functional Brain Imaging Systems," offered by the Department of Biomedical Engineering at K. N. Toosi University of Technology (KNTU).

These assignments were designed to provide students with hands-on experience in processing and analyzing electroencephalography (EEG) data using Python. They were developed by Mohammadreza Shahsavari during his role as a Teaching Assistant for the course.

---

##  Assignments Overview

The repository contains three major assignments that build upon each other, guiding students from basic preprocessing to advanced group-level analysis.

### **Assignment 3: EEG Preprocessing and Visualization**
* **Objective:** To introduce the fundamental pipeline for cleaning and preparing EEG data.
* **Tasks:**
    * Working with EEG data from a goalkeeper participating in a penalty shootout.
    * Performing essential preprocessing steps including filtering, epoching, and baseline correction.
    * Identifying and interpolating bad channels to handle artifacts.
    * Visualizing brain activity by creating animated topomaps of scalp voltage.
* **Key Libraries:** MNE-Python, NumPy, Matplotlib.

### **Assignment 5: Event-Related Potential (ERP) Analysis**
* **Objective:** To analyze ERPs, a critical tool for cognitive neuroscience research.
* **Tasks:**
    * Analyzing EEG data from a visual oddball paradigm to isolate specific cognitive responses.
    * Calculating and plotting the P300 component, a well-known ERP associated with stimulus evaluation.
    * Conducting statistical t-tests to identify electrodes showing a significant P300 effect.
* **Key Libraries:** MNE-Python, SciPy (for statistics), Matplotlib.

### **Final Project: Group-Level ERP Investigation**
* **Objective:** To apply and extend previously learned skills to a complete group-level research question.
* **Tasks:**
    * Performing a comprehensive ERP analysis on a dataset with multiple subjects.
    * Investigating the P300 component's response to target vs. non-target stimuli across a group.
    * Implementing robust preprocessing and artifact rejection pipelines for multiple recordings.
    * Generating grand-average ERP plots and visualizations to report group-level findings.

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
pip install mne numpy scipy pandas matplotlib