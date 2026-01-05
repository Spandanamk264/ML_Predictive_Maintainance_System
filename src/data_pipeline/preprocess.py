"""
Data Preprocessing Pipeline for Predictive Maintenance
=======================================================
Enterprise-grade data pipeline for IoT sensor data preprocessing.

This module handles:
- Loading raw CSV sensor logs
- Schema and dtype validation
- Timestamp generation and alignment
- Missing value handling (interpolation, forward/backward fill)
- Outlier detection and correction
- No data leakage guarantees

Author: IoT Predictive Maintenance Team
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATASET_CONFIG, FEATURE_CONFIG, MODEL_CONFIG,
    get_raw_data_path, get_processed_data_path, MODELS_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validates data schema and dtypes for sensor data."""
    
    # Expected schema definition
    EXPECTED_SCHEMA = {
        'UDI': 'int64',
        'Product ID': 'object',
        'Type': 'object',
        'Air temperature [K]': 'float64',
        'Process temperature [K]': 'float64',
        'Rotational speed [rpm]': 'int64',
        'Torque [Nm]': 'float64',
        'Tool wear [min]': 'int64',
        'Machine failure': 'int64',
        'TWF': 'int64',
        'HDF': 'int64',
        'PWF': 'int64',
        'OSF': 'int64',
        'RNF': 'int64'
    }
    
    # Required columns for modeling
    REQUIRED_COLUMNS = [
        'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
        'Machine failure'
    ]
    
    @classmethod
    def validate_schema(cls, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame schema against expected schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check required columns
        missing_cols = set(cls.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check dtypes for existing columns
        for col, expected_dtype in cls.EXPECTED_SCHEMA.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                # Allow some flexibility in numeric types
                if expected_dtype in ['int64', 'float64']:
                    if not np.issubdtype(df[col].dtype, np.number):
                        errors.append(f"Column '{col}' expected numeric, got {actual_dtype}")
                elif expected_dtype == 'object':
                    if df[col].dtype != 'object':
                        # Try to convert
                        try:
                            df[col] = df[col].astype(str)
                        except Exception as e:
                            errors.append(f"Column '{col}' cannot be converted to string: {e}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @classmethod
    def validate_dtypes(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert columns to expected dtypes.
        
        Args:
            df: DataFrame to convert
            
        Returns:
            DataFrame with corrected dtypes
        """
        df = df.copy()
        
        # Convert numeric columns
        numeric_cols = ['Air temperature [K]', 'Process temperature [K]',
                       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert target to int
        if 'Machine failure' in df.columns:
            df['Machine failure'] = df['Machine failure'].astype(int)
        
        return df


class DataQualityChecker:
    """Checks and corrects data quality issues."""
    
    @staticmethod
    def check_value_ranges(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate sensor readings are within acceptable ranges.
        
        Args:
            df: DataFrame with sensor data
            
        Returns:
            DataFrame with corrected/filtered values
        """
        df = df.copy()
        initial_len = len(df)
        config = FEATURE_CONFIG
        
        # Temperature validation (Kelvin should be > 0)
        if 'Air temperature [K]' in df.columns:
            invalid_mask = (df['Air temperature [K]'] < config.min_temperature_kelvin) | \
                          (df['Air temperature [K]'] > config.max_temperature_kelvin)
            if invalid_mask.sum() > 0:
                logger.warning(f"Found {invalid_mask.sum()} invalid Air temperature readings")
                df.loc[invalid_mask, 'Air temperature [K]'] = np.nan
        
        if 'Process temperature [K]' in df.columns:
            invalid_mask = (df['Process temperature [K]'] < config.min_temperature_kelvin) | \
                          (df['Process temperature [K]'] > config.max_temperature_kelvin)
            if invalid_mask.sum() > 0:
                logger.warning(f"Found {invalid_mask.sum()} invalid Process temperature readings")
                df.loc[invalid_mask, 'Process temperature [K]'] = np.nan
        
        # RPM validation (should be positive)
        if 'Rotational speed [rpm]' in df.columns:
            invalid_mask = (df['Rotational speed [rpm]'] < config.min_rpm) | \
                          (df['Rotational speed [rpm]'] > config.max_rpm)
            if invalid_mask.sum() > 0:
                logger.warning(f"Found {invalid_mask.sum()} invalid RPM readings")
                df.loc[invalid_mask, 'Rotational speed [rpm]'] = np.nan
        
        # Torque validation
        if 'Torque [Nm]' in df.columns:
            invalid_mask = (df['Torque [Nm]'] < config.min_torque) | \
                          (df['Torque [Nm]'] > config.max_torque)
            if invalid_mask.sum() > 0:
                logger.warning(f"Found {invalid_mask.sum()} invalid Torque readings")
                df.loc[invalid_mask, 'Torque [Nm]'] = np.nan
        
        # Tool wear validation
        if 'Tool wear [min]' in df.columns:
            invalid_mask = (df['Tool wear [min]'] < config.min_tool_wear) | \
                          (df['Tool wear [min]'] > config.max_tool_wear)
            if invalid_mask.sum() > 0:
                logger.warning(f"Found {invalid_mask.sum()} invalid Tool wear readings")
                df.loc[invalid_mask, 'Tool wear [min]'] = np.nan
        
        return df

    @staticmethod
    def detect_outliers_iqr(series: pd.Series, multiplier: float = 3.0) -> pd.Series:
        """
        Detect outliers using IQR method.
        
        Args:
            series: Numeric series to check
            multiplier: IQR multiplier for outlier threshold
            
        Returns:
            Boolean mask of outliers
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return (series < lower_bound) | (series > upper_bound)


class DataPipeline:
    """
    Main data preprocessing pipeline for IoT sensor data.
    
    Attributes:
        raw_data_path: Path to raw CSV file
        processed_data_path: Path to save processed data
        df: Loaded DataFrame
        target_col: Name of target column
    """
    
    def __init__(
        self,
        raw_data_path: Optional[str] = None,
        processed_data_path: Optional[str] = None
    ):
        """
        Initialize the data pipeline.
        
        Args:
            raw_data_path: Path to raw CSV sensor logs
            processed_data_path: Directory to save processed data
        """
        self.raw_data_path = raw_data_path or str(get_raw_data_path())
        self.processed_data_path = processed_data_path or str(get_processed_data_path().parent)
        self.df = None
        self.target_col = DATASET_CONFIG.target_column
        self.sensor_cols = DATASET_CONFIG.sensor_columns
        self._scaler = None
        self._label_encoder = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load raw CSV sensor logs with validation.
        
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {self.raw_data_path}...")
        
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Data file not found: {self.raw_data_path}")
        
        self.df = pd.read_csv(self.raw_data_path)
        logger.info(f"Data loaded. Shape: {self.df.shape}")
        
        # Validate schema
        is_valid, errors = SchemaValidator.validate_schema(self.df)
        if not is_valid:
            for error in errors:
                logger.warning(f"Schema validation: {error}")
        
        # Convert dtypes
        self.df = SchemaValidator.validate_dtypes(self.df)
        
        return self.df
    
    def generate_timestamps(self) -> pd.DataFrame:
        """
        Generate a timestamp index for temporal alignment.
        
        Returns:
            DataFrame with timestamp index
        """
        logger.info("Generating timestamp index...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        start_time = pd.Timestamp(FEATURE_CONFIG.timestamp_start)
        interval_minutes = FEATURE_CONFIG.timestamp_interval_minutes
        
        timestamps = pd.date_range(
            start=start_time,
            periods=len(self.df),
            freq=f'{interval_minutes}min'
        )
        
        self.df['timestamp'] = timestamps
        self.df.set_index('timestamp', inplace=True)
        
        logger.info(f"Timestamp index generated: {timestamps[0]} to {timestamps[-1]}")
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """
        Handle missing values and corrupt readings.
        
        Uses time-based interpolation for numeric columns,
        with forward/backward fill for remaining gaps.
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        initial_len = len(self.df)
        
        # Drop irrelevant columns
        cols_to_drop = ['UDI', 'Product ID']
        self.df = self.df.drop(
            columns=[c for c in cols_to_drop if c in self.df.columns],
            errors='ignore'
        )
        
        # Validate value ranges (sets invalid values to NaN)
        self.df = DataQualityChecker.check_value_ranges(self.df)
        
        # Handle missing values
        missing_before = self.df.isnull().sum().sum()
        if missing_before > 0:
            logger.info(f"Handling {missing_before} missing values...")
            
            # Time-based interpolation for numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].interpolate(method='time')
            
            # Forward fill for remaining gaps
            self.df = self.df.ffill()
            
            # Backward fill for any remaining (at the start)
            self.df = self.df.bfill()
            
            missing_after = self.df.isnull().sum().sum()
            logger.info(f"Missing values after cleaning: {missing_after}")
        
        # Final validation - drop rows with any remaining NaN
        self.df = self.df.dropna()
        
        logger.info(f"Data cleaning complete. Rows: {initial_len} -> {len(self.df)}")
        return self.df
    
    def feature_engineering(self) -> pd.DataFrame:
        """
        Create comprehensive feature matrix with lag, rolling, and derived features.
        
        Implements:
        - Lag features (t-1, t-2, t-3)
        - Rolling window statistics (1h, 4h, 8h)
        - Exponential moving averages
        - Rate of change features
        - Domain-specific interaction features
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        sensor_cols = self.sensor_cols
        
        # 1. LAG FEATURES (t-1, t-2, t-3)
        # These use only past data, no leakage
        logger.info("Creating lag features...")
        for col in sensor_cols:
            for lag in FEATURE_CONFIG.lag_steps:
                self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
        
        # 2. ROLLING WINDOW FEATURES
        # Using closed='left' to exclude current observation (no leakage)
        logger.info("Creating rolling window features...")
        for col in sensor_cols:
            for label, window in FEATURE_CONFIG.rolling_windows.items():
                # Shift by 1 to ensure we only use past data
                shifted = self.df[col].shift(1)
                
                for stat in FEATURE_CONFIG.rolling_stats:
                    if stat == 'mean':
                        self.df[f'{col}_roll_mean_{label}'] = shifted.rolling(window=window).mean()
                    elif stat == 'std':
                        self.df[f'{col}_roll_std_{label}'] = shifted.rolling(window=window).std()
                    elif stat == 'min':
                        self.df[f'{col}_roll_min_{label}'] = shifted.rolling(window=window).min()
                    elif stat == 'max':
                        self.df[f'{col}_roll_max_{label}'] = shifted.rolling(window=window).max()
        
        # 3. EXPONENTIAL MOVING AVERAGES
        logger.info("Creating EMA features...")
        for col in sensor_cols:
            for span in FEATURE_CONFIG.ema_spans:
                # Shift by 1 to ensure no leakage
                self.df[f'{col}_ema_{span}'] = self.df[col].shift(1).ewm(span=span).mean()
        
        # 4. RATE OF CHANGE FEATURES
        logger.info("Creating rate-of-change features...")
        for col in sensor_cols:
            self.df[f'{col}_roc'] = self.df[col].pct_change()
        
        # 5. DOMAIN-SPECIFIC INTERACTION FEATURES
        logger.info("Creating interaction features...")
        
        # Temperature difference (heat dissipation indicator)
        self.df['Temp_diff'] = self.df['Process temperature [K]'] - self.df['Air temperature [K]']
        
        # Power (Torque × RPM, proportional to mechanical power)
        self.df['Power'] = self.df['Torque [Nm]'] * self.df['Rotational speed [rpm]']
        
        # Wear rate (change in tool wear over time)
        self.df['Wear_rate'] = self.df['Tool wear [min]'].diff()
        
        # Power per unit RPM (load indicator)
        self.df['Power_per_rpm'] = self.df['Torque [Nm]']  # Torque is essentially this
        
        # Temperature stress (combined thermal load)
        self.df['Temp_stress'] = self.df['Process temperature [K]'] * self.df['Temp_diff']
        
        # 6. ENCODE CATEGORICAL FEATURES
        if 'Type' in self.df.columns:
            logger.info("Encoding categorical features...")
            self._label_encoder = LabelEncoder()
            self.df['Type_encoded'] = self._label_encoder.fit_transform(self.df['Type'])
        
        # Drop rows with NaN created by lagging/rolling operations
        original_len = len(self.df)
        self.df = self.df.dropna()
        
        # Replace infinite values
        self.df = self.df.replace([np.inf, -np.inf], np.nan).dropna()
        
        logger.info(f"Feature engineering complete. Features: {len(self.df.columns)}")
        logger.info(f"Dropped {original_len - len(self.df)} rows due to windowing operations")
        
        return self.df
    
    def generate_failure_target(
        self,
        leading_window_hours: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate binary failure target with configurable leading window.
        
        If the original target indicates failure at time t, this method can
        create a target that marks the preceding N hours as "approaching failure".
        
        Args:
            leading_window_hours: Hours before failure to mark as target=1
            
        Returns:
            DataFrame with adjusted target
        """
        if leading_window_hours is None:
            leading_window_hours = FEATURE_CONFIG.failure_leading_window_hours
        
        if self.df is None or self.target_col not in self.df.columns:
            logger.warning("Cannot generate failure target: data not loaded or target missing")
            return self.df
        
        # Calculate number of observations in the leading window
        observations_per_hour = 60 / FEATURE_CONFIG.timestamp_interval_minutes
        leading_window_obs = int(leading_window_hours * observations_per_hour)
        
        # Create a copy of the target
        original_target = self.df[self.target_col].copy()
        
        # For each failure, mark the preceding N observations
        failure_indices = original_target[original_target == 1].index
        
        new_target = original_target.copy()
        for failure_time in failure_indices:
            # Get the position of this failure
            pos = self.df.index.get_loc(failure_time)
            # Mark preceding observations
            start_pos = max(0, pos - leading_window_obs)
            for i in range(start_pos, pos):
                new_target.iloc[i] = 1
        
        self.df[f'{self.target_col}_lead_{leading_window_hours}h'] = new_target
        
        logger.info(f"Created leading target with {leading_window_hours}h window")
        logger.info(f"Original failures: {original_target.sum()}, "
                   f"New target positives: {new_target.sum()}")
        
        return self.df
    
    def get_train_test_split(
        self,
        test_size: float = 0.2,
        shuffle: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Get time-based train/test split (no shuffling for time series).
        
        Args:
            test_size: Proportion of data for testing
            shuffle: Whether to shuffle (should be False for time series)
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        # Get feature columns
        feature_cols = [c for c in self.df.columns 
                       if c not in DATASET_CONFIG.exclude_columns]
        
        X = self.df[feature_cols]
        y = self.df[self.target_col]
        
        if shuffle:
            logger.warning("Using shuffle=True for time series data may cause data leakage!")
        
        # Time-based split (preserve temporal order)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        logger.info(f"Train failure rate: {y_train.mean():.4f}, "
                   f"Test failure rate: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed(self, filename: Optional[str] = None) -> str:
        """
        Save processed data to CSV.
        
        Args:
            filename: Optional filename (uses default from config if not provided)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = DATASET_CONFIG.processed_data_filename
        
        output_path = os.path.join(self.processed_data_path, filename)
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        self.df.to_csv(output_path)
        logger.info(f"Processed data saved to {output_path}")
        
        return output_path
    
    def save_preprocessing_artifacts(self, save_path: Optional[str] = None) -> Dict[str, str]:
        """
        Save preprocessing artifacts (scaler, encoder) for inference.
        
        Args:
            save_path: Directory to save artifacts
            
        Returns:
            Dictionary of artifact paths
        """
        if save_path is None:
            save_path = str(MODELS_DIR)
        
        os.makedirs(save_path, exist_ok=True)
        artifacts = {}
        
        # Save label encoder if available
        if self._label_encoder is not None:
            encoder_path = os.path.join(save_path, 'label_encoder.joblib')
            joblib.dump(self._label_encoder, encoder_path)
            artifacts['label_encoder'] = encoder_path
        
        # Save feature columns list
        feature_cols = [c for c in self.df.columns 
                       if c not in DATASET_CONFIG.exclude_columns]
        features_path = os.path.join(save_path, 'feature_columns.joblib')
        joblib.dump(feature_cols, features_path)
        artifacts['feature_columns'] = features_path
        
        logger.info(f"Preprocessing artifacts saved to {save_path}")
        return artifacts
    
    def get_summary(self) -> pd.Series:
        """
        Get feature correlation summary with target.
        
        Returns:
            Series of correlations sorted by absolute value
        """
        logger.info("\nFeature Correlation with Target:")
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if self.target_col not in numeric_df.columns:
            logger.warning(f"Target column '{self.target_col}' not in numeric columns")
            return pd.Series()
        
        corr = numeric_df.corr()[self.target_col].sort_values(
            key=abs, ascending=False
        )
        
        logger.info("Top 10 correlated features:")
        print(corr.head(10))
        
        logger.info("\nBottom 5 correlated features:")
        print(corr.tail(5))
        
        return corr


def run_full_pipeline() -> pd.DataFrame:
    """
    Run the complete data preprocessing pipeline.
    
    Returns:
        Processed DataFrame ready for modeling
    """
    logger.info("=" * 60)
    logger.info("RUNNING FULL DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    pipeline = DataPipeline()
    
    # Step 1: Load data
    pipeline.load_data()
    
    # Step 2: Generate timestamps
    pipeline.generate_timestamps()
    
    # Step 3: Clean data
    pipeline.clean_data()
    
    # Step 4: Feature engineering
    pipeline.feature_engineering()
    
    # Step 5: Save processed data
    pipeline.save_processed()
    
    # Step 6: Save preprocessing artifacts
    pipeline.save_preprocessing_artifacts()
    
    # Step 7: Get summary
    pipeline.get_summary()
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Final dataset shape: {pipeline.df.shape}")
    logger.info("=" * 60)
    
    return pipeline.df


if __name__ == "__main__":
    run_full_pipeline()
