# ECG Arrhythmia Detection with a 1D CNN (tested on unseen patients)

A 1D convolutional neural network that classifies individual heartbeats from ECG signals into five AAMI classes (four are evaluated, see below). It is evaluated the honest way: on patients the model never saw during training.

**Stack:** Python, wfdb, SciPy, PyTorch, scikit-learn, Google Colab

## Data
- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/) (PhysioNet): 48 half-hour ECG recordings, every beat annotated by cardiologists, sampled at 360 Hz.
- Four pacemaker records (102, 104, 107, 217) excluded, as the AAMI standard recommends.

## Method
1. Band-pass filter (0.5 to 40 Hz) to remove baseline drift and high-frequency noise.
2. Cut a window around every annotated R-peak (0.25 s before, 0.45 s after = 252 samples). Normalize each record.
3. Map annotation symbols to AAMI classes: N (normal), S (supraventricular), V (ventricular), F (fusion).
4. **Inter-patient split (de Chazal et al.):** train on 22 patients (DS1), test on 22 different patients (DS2). No patient appears in both sets.
5. Baseline: random forest on RR-interval (timing) features only.
6. Model: 1D CNN over the beat waveform, with 6 RR-interval features joined at the final layer. Class-weighted loss because most beats are normal.
7. Number of epochs chosen by patient-grouped 5-fold cross-validation on DS1 only. DS2 was used only for final testing.

## Results (DS2, 22 unseen patients)
Macro F1 is over N, S, V, F. CNN numbers are mean ± std over 3 random seeds, 15 epochs.

| Model | Macro F1 | V recall |
|---|---|---|
| Random forest on RR intervals | 0.389 | 0.606 |
| **1D CNN (waveform + RR)** | **0.464 ± 0.044** | **0.963 ± 0.016** |
| 1D CNN without RR features (ablation) | 0.430 ± 0.031 | n/a |

Per-class results for the median seed (macro F1 0.474):

| Class | Precision | Recall | Beats |
|---|---|---|---|
| N | 0.986 | 0.833 | 44,240 |
| S | 0.243 | 0.626 | 1,837 |
| V | 0.480 | 0.982 | 3,220 |
| F | 0.000 | 0.000 | 388 |

![Confusion matrix](confusion_matrix.png)
![ECG strip](ecg_strip.png)

*The strip shows record 200, a patient where the model does well. It is a good example, not an average one.*

## Honest limitations
- **Ventricular (V) beats are found reliably, but at a cost:** recall is high (about 96%) but precision is only about 48%, so roughly half of the V alarms are false.
- **Fusion (F) beats are not detected** (0 of 388). Macro F1 includes this zero.
- **Supraventricular (S) beats are only moderately detected** (recall 0.63, precision 0.24). About 75% of the S beats in the test set come from a single patient (record 232), so this number is not very robust.
- **The CNN's advantage over the baseline is modest.** All 3 seeds beat the baseline, but the gap (0.075 macro F1) is under 2 standard deviations. Removing the RR features gives 0.430 ± 0.031, which is within seed noise of 0.464, so these experiments do not show that the RR features help the CNN.
- **Only 3 seeds**, so treat the standard deviations as rough.
- Results are well below the 98 to 99% figures common online. Those come from splitting beats randomly, which leaks each patient's beats into both train and test sets.
- Record 114 has its two ECG leads in swapped order; the first lead is used for all records.

## Development notes (full disclosure)
- The Q (unknown) class has only 8 training and 7 test beats, so it is excluded from training and evaluation.
- An earlier version of the pipeline (with Q, simpler RR features, a small held-out validation set) was run on DS2 and scored 0.452 macro F1 (N, S, V, F). The pipeline was then revised: Q dropped, 6 RR features, cross-validation on DS1 for model selection.
- The revised pipeline was first run with a default of 10 epochs (0.515 ± 0.036 macro F1). DS1 cross-validation indicated 15 epochs (CV macro F1 0.427 vs 0.362), so the final results use 15 epochs. Both runs are reported here.
- Normalizing each beat on its own (instead of each record) scored 0.37 in DS1 cross-validation, not better than per-record normalization, and was not used.

## How to run
1. Open the notebook in Google Colab (GPU runtime recommended).
2. Run the cells in order. The first run downloads the MIT-BIH records to Google Drive (a few minutes); later runs reuse them.
