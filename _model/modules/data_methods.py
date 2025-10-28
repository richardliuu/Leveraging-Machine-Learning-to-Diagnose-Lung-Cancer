import pandas as pd 

class DataHandling:
    """
    The DataHandling Class handles and transforms the Random Forest performance data into training data. 

    The instantiated variables for training and testing are managed through class functions.
    Data Splits are handled in split().

    This class encapsulates all data preprocessing steps including:
    - Data integrity checks for duplicate samples and label consistency
    - Feature selection and column dropping
    - Patient-grouped cross-validation splits
    - Class distribution analysis

    Attributes:
        data (str): Path to the input CSV file
        
        Storage for cross-validation results:
        - reports: Classification reports from each fold
        - conf_matrices: Confusion matrices from each fold
        - roc_aucs: ROC-AUC scores from each fold
        - fold_details: Fold-specific statistics and metadata
        - predictions: Model predictions from each fold
    """ 
    def __init__(self):
        self.data = "data/train_data"
        
        # Storage for cross-validation results
        self.reports = []
        self.conf_matrices = []
        self.roc_aucs = []
        self.fold_details = []
        self.predictions = []
        
        # Data attributes
        self.X = None
        self.y = None
        self.groups = None
        self.feature_cols = None
        
        # Current fold data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_patients = None
        self.test_patients = None
        
    def load_data(self):
        """
        Load and perform comprehensive data integrity checks on the lung cancer dataset.
        
        This method loads the dataset, extracts features and labels, and validates data quality
        by checking for duplicate samples and ensuring patient-label consistency across all samples.
        It also provides a summary of the class distribution.
        
        Returns:
            bool: True if data passes all integrity checks (no duplicates and consistent 
                  patient labels), False otherwise.
        
        Checks Performed:
            1. Duplicate Detection: Identifies samples with identical feature values
            2. Patient-Label Consistency: Ensures each patient has consistent cancer stage labels
            3. Class Distribution: Reports overall distribution of cancer stages
        
        Side Effects:
            - Prints data loading statistics and warnings for any integrity issues found
            - Sets class attributes: X, y, feature_cols, groups
        
        Data Structure Expected:
            CSV file with columns: 'chunk', 'cancer_stage', 'patient_id', 'filename', 
            'rolloff', 'bandwidth', 'skew', 'zcr', 'rms', plus feature columns
        
        Example:
            >>> handler = DataHandling()
            >>> is_clean = handler.load_data()
            >>> if is_clean:
            ...     print("Data passed all integrity checks")
        """
        print("Loading dataset...")
        df = pd.read_csv(self.data)
        print(f"Loaded {len(df)} samples from {df['patient_id'].nunique()} patients")
        
        # Prepare features and labels - drop metadata columns
        self.X = df.drop(columns=['chunk', 'cancer_stage', 'patient_id', 'filename', 
                                 'rolloff', 'bandwidth', "skew", "zcr", 'rms'])
        self.y = df['cancer_stage']
        self.groups = df['patient_id']
        self.feature_cols = self.X.columns.tolist()
        
        print(f"Total samples: {len(df)}")
        print(f"Total patients: {df['patient_id'].nunique()}")
        print(f"Features: {self.X.shape[1]}")
        
        # Perform data integrity verification
        return self._verify_data_integrity(df)
    
    def _verify_data_integrity(self, df):
        """
        Verify data integrity by checking for duplicates and inconsistent patient labels.
        
        Args:
            df (pd.DataFrame): The loaded dataset
            
        Returns:
            bool: True if data passes all checks, False otherwise
        
        Private method that performs:
            - Duplicate sample detection across feature columns
            - Patient label consistency verification
            - Class distribution analysis and reporting
        """
        
        print("\n=== DATA INTEGRITY CHECKS ===")
        
        # Check for duplicate samples
        duplicates = df.duplicated(subset=self.feature_cols)
        print(f"Duplicate feature rows: {duplicates.sum()}")
        
        if duplicates.sum() > 0:
            print("WARNING: Duplicate samples found!")
            dup_rows = df[duplicates]
            print(f"Example duplicate patients: {dup_rows['patient_id'].unique()[:5]}")
        else:
            print("No duplicate feature rows found")
        
        # Check for inconsistent patient labels
        patient_labels = df.groupby('patient_id')['cancer_stage'].nunique()
        inconsistent_patients = patient_labels[patient_labels > 1]
        if len(inconsistent_patients) > 0:
            print(f"WARNING: {len(inconsistent_patients)} patients have inconsistent labels!")
            print("Inconsistent patients:", inconsistent_patients.index.tolist()[:10])
        else:
            print("All patients have consistent labels")
        
        # Display class distribution
        print(f"\nOverall class distribution:")
        class_counts = df['cancer_stage'].value_counts()
        print(class_counts)
        print(f"Class ratio: {class_counts.iloc[0]/class_counts.iloc[1]:.2f}:1")
        
        return duplicates.sum() == 0 and len(inconsistent_patients) == 0
    
    def split(self, df, train_idx, test_idx):
        """
        Split the dataset into training and testing sets for the current fold.
        
        Args:
            df (pd.DataFrame): The complete dataset
            train_idx (np.ndarray): Indices for training samples
            test_idx (np.ndarray): Indices for testing samples
        
        Side Effects:
            Sets the following class attributes:
            - X_train, X_test: Training and testing feature matrices
            - y_train, y_test: Training and testing labels
            - train_patients, test_patients: Sets of patient IDs for each split
        
        This method handles the data splitting for GroupKFold cross-validation,
        ensuring that patient grouping is maintained.
        """
        self.X_train, self.X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
        self.y_train, self.y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
        
        # Track patients in each split to verify no leakage
        self.train_patients = set(df.iloc[train_idx]['patient_id'])
        self.test_patients = set(df.iloc[test_idx]['patient_id'])
