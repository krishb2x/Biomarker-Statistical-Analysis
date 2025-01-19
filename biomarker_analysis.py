import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from scipy import stats
import time

class CoverageAnalysis:
    def __init__(self, input_file):
        self.df = pd.read_csv(input_file)
        self.coverage_columns = [col for col in self.df.columns if col.strip('`').isdigit()]

    def calculate_cpg_coverage(self):
        # Calculate CpG Coverage by summing over the coverage columns (vectorized)
        self.df['CpG_Coverage'] = self.df[self.coverage_columns].sum(axis=1)

    def calculate_statistics(self):
        # Efficient calculation of statistics: median, mean, std, CV, range, skewness, kurtosis
        coverage_stats = self.df.groupby('Tissue')['CpG_Coverage'].agg(
            median='median',
            mean='mean',
            std='std'
        ).reset_index()

        coverage_stats['CV'] = coverage_stats['std'] / coverage_stats['mean'] * 100  # CV in percentage
        coverage_stats['range'] = coverage_stats['mean'] + coverage_stats['std'] - (coverage_stats['mean'] - coverage_stats['std'])  # Coverage range
        coverage_stats['skewness'] = self.df.groupby('Tissue')['CpG_Coverage'].apply(lambda x: stats.skew(x)).reset_index(drop=True)  # Skewness
        coverage_stats['kurtosis'] = self.df.groupby('Tissue')['CpG_Coverage'].apply(lambda x: stats.kurtosis(x)).reset_index(drop=True)  # Kurtosis
        self.coverage_stats = coverage_stats

        # Save coverage statistics to CSV
        self.coverage_stats.to_csv('coverage_statistics.csv', index=False)

    def calculate_median_and_cv(self):
        # Vectorized calculation of median and CV for each tissue
        coverage_data = self.df[self.coverage_columns]
        coverage_median = coverage_data.median(axis=1)
        coverage_mean = coverage_data.mean(axis=1)
        coverage_std = coverage_data.std(axis=1)
        coverage_cv = (coverage_std / coverage_mean) * 100

        self.df['median'] = coverage_median
        self.df['CV'] = coverage_cv

        # Save median and CV to CSV
        self.df[['Tissue', 'median', 'CV']].to_csv('median_cv.csv', index=False)

    def plot_and_save_coverage_statistics(self):
        # Generate coverage statistics plots
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot median
        sns.boxplot(x='Tissue', y='median', data=self.df, ax=ax[0])
        ax[0].set_title('Median CpG Coverage by Tissue')

        # Plot CV
        sns.boxplot(x='Tissue', y='CV', data=self.df, ax=ax[1])
        ax[1].set_title('CV of CpG Coverage by Tissue')

        # Save plots
        plt.tight_layout()
        plt.savefig('coverage_statistics.png')
        plt.close(fig)

    def random_forest_classifier(self):
        # Use Random Forest Classifier to identify PMPs with high specificity for tissue differentiation
        X = self.df[self.coverage_columns]  # Using coverage columns as features
        y = self.df['Tissue']  # Target variable (Tissue type)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Predict and evaluate the model
        y_pred = rf.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Feature importance (for identifying PMPs)
        feature_importance = pd.Series(rf.feature_importances_, index=self.coverage_columns).sort_values(ascending=False)

        # Save feature importance to CSV
        feature_importance.to_csv('feature_importance.csv', header=['Importance'])

        # Save feature importance plot
        plt.figure(figsize=(10, 6))
        feature_importance.plot(kind='bar', color='skyblue')
        plt.title('Feature Importance (Top PMPs)')
        plt.xlabel('Coverage Columns')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

        return rf, X_test, y_test

    def calculate_vrf(self):
        # Calculate the Variant Read Fraction (VRF) for each PMP
        self.df['VRF'] = self.df[self.coverage_columns].sum(axis=1) / self.df[self.coverage_columns].count(axis=1)

        # Save VRF to CSV
        self.df[['Tissue', 'VRF']].to_csv('variant_read_fraction.csv', index=False)

    def plot_and_save_roc_curve(self, rf, X_test, y_test):
        # ROC analysis for Tissue differentiation
        fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1], pos_label="Tissue_2")
        roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.tight_layout()

        # Save ROC curve plot
        plt.savefig('roc_curve.png')
        plt.close()

    def analyze_sequencing_depth(self):
        # Analyze how sequencing depth affects specificity confidence
        depth = 1000000  # Sequencing depth
        pmp_thresholds = self.coverage_stats['mean'] * 0.1  # 10% of mean coverage
        self.coverage_stats['pmp_threshold'] = pmp_thresholds
        self.coverage_stats['confidence_at_depth'] = pmp_thresholds / depth  # Simplified confidence estimate

        # Save depth analysis to CSV
        self.coverage_stats[['Tissue', 'pmp_threshold', 'confidence_at_depth']].to_csv('sequencing_depth_analysis.csv', index=False)

    def estimate_reads_for_tissue(self):
        # Estimate the threshold of reads required to confidently call Tissue #2 at a sequencing depth of 1 million reads
        tissue_2_mean = self.coverage_stats.loc[self.coverage_stats['Tissue'] == 'Tissue_2', 'mean'].values[0]
        reads_needed = tissue_2_mean / 0.1  # Assuming 10% of mean coverage is sufficient for confident calling
        self.coverage_stats['reads_needed_for_tissue_2'] = reads_needed

        # Save reads estimation to CSV
        self.coverage_stats[['Tissue', 'reads_needed_for_tissue_2']].to_csv('reads_estimation.csv', index=False)

    def compare_specificity_with_cpg(self):
        # Compare the specificity of the top 10 PMPs against individual CpG sites
        top_pmp_features = self.coverage_stats.nlargest(10, 'mean')['Tissue'].values
        individual_cpg_comparison = self.df[self.df['Tissue'].isin(top_pmp_features)].groupby('Tissue').mean()
        self.spec_specificity_comparison = individual_cpg_comparison

        # Save specificity comparison to CSV
        self.spec_specificity_comparison.to_csv('specificity_comparison.csv', index=True)

# Main execution
if __name__ == "__main__":
    start_time = time.time()  # Track execution time
    analysis = CoverageAnalysis('/mnt/India2/bioinfo2/PGTA/Nanopore/KrishnaWdir/TEST/input.csv')
    analysis.calculate_cpg_coverage()
    analysis.calculate_statistics()
    analysis.calculate_median_and_cv()
    analysis.plot_and_save_coverage_statistics()
    rf, X_test, y_test = analysis.random_forest_classifier()
    analysis.calculate_vrf()
    analysis.plot_and_save_roc_curve(rf, X_test, y_test)
    analysis.analyze_sequencing_depth()
    analysis.estimate_reads_for_tissue()
    analysis.compare_specificity_with_cpg()

    end_time = time.time()  # Track execution time
    print(f"Script executed in {end_time - start_time:.2f} seconds.")
