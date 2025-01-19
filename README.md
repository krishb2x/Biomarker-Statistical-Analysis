# Biomarker-Statistical-Analysis

Hello, I developed this python script as part of an interview assignment to performs statistical analysis. It follows specific assignment protocols rather than adhering to clinical standards. 
1. Calculate coverage statistics (median, mean, CV, etc.).
2. Identify potential biomarkers (PMPs) for tissue differentiation using a Random Forest classifier.
3. Calculate Variant Read Fraction (VRF) for each PMP.
4. Analyze the effect of sequencing depth on specificity.
5. Estimate the required reads to confidently call tissue at a given depth.
6. Compare the specificity of top biomarkers against individual CpG sites.

# Requirements
1. Python 3.x
2. lib: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy

# How to Run
1. Place your input.csv file in the same directory as the script.
2. Run the script:
 python biomarker_analysis.py

# Output
1. CSV Files: Includes coverage statistics, feature importance, VRF, sequencing depth analysis, and specificity comparison.
2. Images: Boxplots, feature importance chart, and ROC curve plot.
