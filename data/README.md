# CWRU Bearing Dataset

This folder is reserved for the **Case Western Reserve University (CWRU) Bearing Fault Dataset** used in this project.

## Important Notice

The original `.mat` files are **not included in this repository** because:

- The dataset files are large in size
- Keeping raw data out of GitHub makes the repository cleaner and faster
- Users should download data directly from the official source

## Official Source

Download the dataset from the Case Western Reserve University Bearing Data Center:

🔗 [https://engineering.case.edu/bearingdatacenter](https://engineering.case.edu/bearingdatacenter/download-data-file)

## Required Data for This Project

Please download the **12 kHz Drive End Bearing Fault Data**.

Ensure the following operating conditions are included:

### Load Conditions
- 0 HP
- 1 HP
- 2 HP
- 3 HP

### Classes Needed
- Normal
- Inner Race Fault (IR)
- Outer Race Fault (OR)
- Ball Fault (B)

### Fault Sizes Required
Use these defect diameters:

- 0.007 in
- 0.014 in
- 0.021 in

> Note: `0.028 in` faults are intentionally excluded because they are not consistently available across all load conditions.

## Recommended Folder Structure

After downloading, place the `.mat` files inside this folder:

```text
data/
├── Normal_0.mat
├── IR_7_load0.mat
├── OR_14_load1.mat
├── B_21_load3.mat
└── ...
