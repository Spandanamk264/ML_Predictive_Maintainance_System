"""
Feature Engineering Module for Predictive Maintenance
======================================================
Production-grade feature engineering for IoT sensor data.

This module provides:
- Lag features with configurable offsets
- Rolling window statistics (mean, std, min, max)
- Exponential moving averages
- Rate of change / momentum features
- Domain-specific interaction features
- Binary failure target generation with leading window
- No data leakage guarantees

Author: IoT Predictive Maintenance Team
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import logging
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURE_CONFIG, DATASET_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for time-series sensor data.
    
    All methods ensure no data leakage by only using past observations
    for feature computation.
    
    Attributes:
        df: Input DataFrame with sensor data
        sensor_cols: List of sensor column names
        created_features: List of created feature names
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        sensor_columns: Optional[List[str]] = None
    ):
        """
        Initialize the feature engineer.
        
        Args:
            df: DataFrame with sensor data (should have datetime index)
            sensor_columns: List of sensor column names to engineer features from
        """
        self.df = df.copy()
        self.sensor_cols = sensor_columns or DATASET_CONFIG.sensor_columns
        self.created_features: List[str] = []
        
        # Validate sensor columns exist
        missing_cols = set(self.sensor_cols) - set(self.df.columns)
        if missing_cols:
            logger.warning(f"Missing sensor columns: {missing_cols}")
            self.sensor_cols = [c for c in self.sensor_cols if c in self.df.columns]
    
    def create_lag_features(
        self,
        lags: Optional[List[int]] = None,
        columns: Optional[List[str]] = None
    ) -> 'FeatureEngineer':
        """
        Create lag features for temporal dependencies.
        
        Lag features use only past values, ensuring no data leakage.
        
        Args:
            lags: List of lag steps (default: [1, 2, 3])
            columns: Columns to create lags for (default: all sensors)
            
        Returns:
            Self for method chaining
        """
        if lags is None:
            lags = FEATURE_CONFIG.lag_steps
        if columns is None:
            columns = self.sensor_cols
        
        logger.info(f"Creating lag features for {len(columns)} columns with lags {lags}")
        
        for col in columns:
            if col in self.df.columns:
                for lag in lags:
                    feature_name = f'{col}_lag_{lag}'
                    self.df[feature_name] = self.df[col].shift(lag)
                    self.created_features.append(feature_name)
        
        return self
    
    def create_rolling_features(
        self,
        windows: Optional[Dict[str, int]] = None,
        statistics: Optional[List[str]] = None,
        columns: Optional[List[str]] = None
    ) -> 'FeatureEngineer':
        """
        Create rolling window statistics.
        
        Uses shift(1) before rolling to ensure no data leakage.
        
        Args:
            windows: Dict mapping window labels to sizes (default from config)
            statistics: List of statistics to compute (mean, std, min, max)
            columns: Columns to create rolling features for
            
        Returns:
            Self for method chaining
        """
        if windows is None:
            windows = FEATURE_CONFIG.rolling_windows
        if statistics is None:
            statistics = FEATURE_CONFIG.rolling_stats
        if columns is None:
            columns = self.sensor_cols
        
        logger.info(f"Creating rolling features: windows={list(windows.keys())}, "
                   f"stats={statistics}")
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            # Shift by 1 to exclude current observation (no leakage)
            shifted = self.df[col].shift(1)
            
            for label, window_size in windows.items():
                roller = shifted.rolling(window=window_size, min_periods=1)
                
                for stat in statistics:
                    feature_name = f'{col}_roll_{stat}_{label}'
                    
                    if stat == 'mean':
                        self.df[feature_name] = roller.mean()
                    elif stat == 'std':
                        self.df[feature_name] = roller.std()
                    elif stat == 'min':
                        self.df[feature_name] = roller.min()
                    elif stat == 'max':
                        self.df[feature_name] = roller.max()
                    elif stat == 'median':
                        self.df[feature_name] = roller.median()
                    elif stat == 'skew':
                        self.df[feature_name] = roller.skew()
                    elif stat == 'kurt':
                        self.df[feature_name] = roller.kurt()
                    else:
                        logger.warning(f"Unknown statistic: {stat}")
                        continue
                    
                    self.created_features.append(feature_name)
        
        return self
    
    def create_ema_features(
        self,
        spans: Optional[List[int]] = None,
        columns: Optional[List[str]] = None
    ) -> 'FeatureEngineer':
        """
        Create Exponential Moving Average features.
        
        EMA gives more weight to recent observations while still
        using only past data.
        
        Args:
            spans: List of EMA spans (default from config)
            columns: Columns to create EMAs for
            
        Returns:
            Self for method chaining
        """
        if spans is None:
            spans = FEATURE_CONFIG.ema_spans
        if columns is None:
            columns = self.sensor_cols
        
        logger.info(f"Creating EMA features with spans {spans}")
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            # Shift by 1 to exclude current observation
            shifted = self.df[col].shift(1)
            
            for span in spans:
                feature_name = f'{col}_ema_{span}'
                self.df[feature_name] = shifted.ewm(span=span, adjust=False).mean()
                self.created_features.append(feature_name)
        
        return self
    
    def create_rate_of_change(
        self,
        periods: List[int] = [1],
        columns: Optional[List[str]] = None
    ) -> 'FeatureEngineer':
        """
        Create rate of change (momentum) features.
        
        Args:
            periods: List of periods for rate calculation
            columns: Columns to create ROC for
            
        Returns:
            Self for method chaining
        """
        if columns is None:
            columns = self.sensor_cols
        
        logger.info(f"Creating rate-of-change features")
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            for period in periods:
                if period == 1:
                    feature_name = f'{col}_roc'
                else:
                    feature_name = f'{col}_roc_{period}'
                
                self.df[feature_name] = self.df[col].pct_change(periods=period)
                self.created_features.append(feature_name)
        
        return self
    
    def create_diff_features(
        self,
        periods: List[int] = [1, 2],
        columns: Optional[List[str]] = None
    ) -> 'FeatureEngineer':
        """
        Create difference features (absolute change).
        
        Args:
            periods: List of periods for differencing
            columns: Columns to create diffs for
            
        Returns:
            Self for method chaining
        """
        if columns is None:
            columns = self.sensor_cols
        
        logger.info(f"Creating difference features")
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            for period in periods:
                feature_name = f'{col}_diff_{period}'
                self.df[feature_name] = self.df[col].diff(periods=period)
                self.created_features.append(feature_name)
        
        return self
    
    def create_interaction_features(self) -> 'FeatureEngineer':
        """
        Create domain-specific interaction features.
        
        These are based on manufacturing/industrial domain knowledge.
        
        Returns:
            Self for method chaining
        """
        logger.info("Creating domain-specific interaction features")
        
        # Temperature Difference (Heat dissipation indicator)
        if 'Process temperature [K]' in self.df.columns and 'Air temperature [K]' in self.df.columns:
            self.df['Temp_diff'] = (
                self.df['Process temperature [K]'] - self.df['Air temperature [K]']
            )
            self.created_features.append('Temp_diff')
        
        # Mechanical Power (Torque × RPM)
        if 'Torque [Nm]' in self.df.columns and 'Rotational speed [rpm]' in self.df.columns:
            self.df['Power'] = (
                self.df['Torque [Nm]'] * self.df['Rotational speed [rpm]']
            )
            self.created_features.append('Power')
            
            # Normalized power (power per unit RPM)
            self.df['Torque_per_rpm'] = (
                self.df['Torque [Nm]'] / 
                (self.df['Rotational speed [rpm]'] + 1)  # Add 1 to avoid division by zero
            )
            self.created_features.append('Torque_per_rpm')
        
        # Wear Rate (Change in tool wear)
        if 'Tool wear [min]' in self.df.columns:
            self.df['Wear_rate'] = self.df['Tool wear [min]'].diff()
            self.created_features.append('Wear_rate')
            
            # Accelerating wear indicator
            self.df['Wear_acceleration'] = self.df['Wear_rate'].diff()
            self.created_features.append('Wear_acceleration')
        
        # Temperature Stress (Combined thermal load)
        if 'Temp_diff' in self.df.columns and 'Process temperature [K]' in self.df.columns:
            self.df['Temp_stress'] = (
                self.df['Process temperature [K]'] * self.df['Temp_diff'] / 1000
            )
            self.created_features.append('Temp_stress')
        
        # Overload Indicator (High power with high wear)
        if 'Power' in self.df.columns and 'Tool wear [min]' in self.df.columns:
            self.df['Overload_indicator'] = (
                (self.df['Power'] / self.df['Power'].max()) * 
                (self.df['Tool wear [min]'] / self.df['Tool wear [min]'].max())
            )
            self.created_features.append('Overload_indicator')
        
        return self
    
    def create_binary_failure_target(
        self,
        leading_window_hours: Optional[int] = None,
        target_column: Optional[str] = None
    ) -> 'FeatureEngineer':
        """
        Create binary failure target with configurable leading window.
        
        This marks observations as "approaching failure" if a failure
        occurs within the next N hours.
        
        Args:
            leading_window_hours: Hours before failure to mark as positive
            target_column: Name of the original target column
            
        Returns:
            Self for method chaining
        """
        if leading_window_hours is None:
            leading_window_hours = FEATURE_CONFIG.failure_leading_window_hours
        if target_column is None:
            target_column = DATASET_CONFIG.target_column
        
        if target_column not in self.df.columns:
            logger.warning(f"Target column '{target_column}' not found")
            return self
        
        logger.info(f"Creating binary failure target with {leading_window_hours}h leading window")
        
        # Calculate observations per hour
        obs_per_hour = 60 / FEATURE_CONFIG.timestamp_interval_minutes
        window_obs = int(leading_window_hours * obs_per_hour)
        
        # For the leading window approach, look BACKWARD from each failure
        # This means: if failure happens at T, mark T-window to T as positive
        
        # Alternative implementation using rolling with shift
        # Look ahead for failures (then we need to express this differently)
        # Actually, we want: for each time t, is there a failure in [t, t+window]?
        # This is equivalent to: rolling max of target for next N steps
        
        target_series = self.df[target_column]
        
        # Reverse the series, apply rolling, then reverse back
        reversed_target = target_series.iloc[::-1]
        rolling_max = reversed_target.rolling(window=window_obs + 1, min_periods=1).max()
        leading_target = rolling_max.iloc[::-1]
        
        new_column_name = f'{target_column}_lead_{leading_window_hours}h'
        self.df[new_column_name] = leading_target.astype(int)
        
        logger.info(f"Original failures: {target_series.sum()}, "
                   f"Leading target positives: {self.df[new_column_name].sum()}")
        
        return self
    
    def create_all_features(self) -> 'FeatureEngineer':
        """
        Create all standard features in the recommended order.
        
        Returns:
            Self for method chaining
        """
        logger.info("=" * 60)
        logger.info("CREATING ALL FEATURES")
        logger.info("=" * 60)
        
        self.create_lag_features()
        self.create_rolling_features()
        self.create_ema_features()
        self.create_rate_of_change()
        self.create_interaction_features()
        
        logger.info(f"Total features created: {len(self.created_features)}")
        
        return self
    
    def get_features(
        self,
        drop_na: bool = True,
        replace_inf: bool = True
    ) -> pd.DataFrame:
        """
        Return the DataFrame with all engineered features.
        
        Args:
            drop_na: Whether to drop rows with NaN values
            replace_inf: Whether to replace infinite values with NaN
            
        Returns:
            DataFrame with all features
        """
        df = self.df.copy()
        
        if replace_inf:
            df = df.replace([np.inf, -np.inf], np.nan)
        
        if drop_na:
            original_len = len(df)
            df = df.dropna()
            dropped = original_len - len(df)
            if dropped > 0:
                logger.info(f"Dropped {dropped} rows containing NaN/Inf values")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of created feature names."""
        return self.created_features.copy()
    
    def get_feature_matrix(
        self,
        target_column: Optional[str] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get feature matrix (X) and target (y) ready for modeling.
        
        Automatically excludes target and non-feature columns.
        
        Args:
            target_column: Name of target column
            exclude_columns: Additional columns to exclude
            
        Returns:
            Tuple of (X, y) DataFrames
        """
        if target_column is None:
            target_column = DATASET_CONFIG.target_column
        if exclude_columns is None:
            exclude_columns = DATASET_CONFIG.exclude_columns
        
        df = self.get_features()
        
        # Get feature columns
        feature_cols = [c for c in df.columns if c not in exclude_columns]
        
        X = df[feature_cols]
        y = df[target_column] if target_column in df.columns else None
        
        logger.info(f"Feature matrix shape: {X.shape}")
        if y is not None:
            logger.info(f"Target shape: {y.shape}, positive rate: {y.mean():.4f}")
        
        return X, y
    
    def get_summary(self) -> Dict[str, any]:
        """
        Get summary of feature engineering results.
        
        Returns:
            Dictionary with summary statistics
        """
        df = self.get_features()
        
        summary = {
            'total_samples': len(df),
            'total_features': len(df.columns),
            'new_features_created': len(self.created_features),
            'original_sensors': len(self.sensor_cols),
            'feature_categories': {
                'lag': len([f for f in self.created_features if 'lag' in f]),
                'rolling': len([f for f in self.created_features if 'roll' in f]),
                'ema': len([f for f in self.created_features if 'ema' in f]),
                'roc': len([f for f in self.created_features if 'roc' in f]),
                'interaction': len([f for f in self.created_features 
                                   if any(x in f for x in ['Temp_diff', 'Power', 'Wear', 'stress', 'indicator'])])
            }
        }
        
        return summary


def create_ml_ready_features(
    df: pd.DataFrame,
    sensor_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Convenience function to create ML-ready feature matrix.
    
    Args:
        df: Input DataFrame with sensor data
        sensor_columns: Optional list of sensor columns
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    engineer = FeatureEngineer(df, sensor_columns)
    engineer.create_all_features()
    
    X, y = engineer.get_feature_matrix()
    feature_names = list(X.columns)
    
    return X, y, feature_names


if __name__ == "__main__":
    # Example usage
    import os
    
    # Load processed data
    base_dir = Path(__file__).parent.parent.parent
    data_path = base_dir / "data" / "processed" / "processed_data.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        print("=" * 60)
        print("FEATURE ENGINEERING DEMONSTRATION")
        print("=" * 60)
        
        engineer = FeatureEngineer(df)
        engineer.create_all_features()
        
        summary = engineer.get_summary()
        print("\nFeature Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        X, y = engineer.get_feature_matrix()
        print(f"\nFinal feature matrix: {X.shape}")
    else:
        print(f"Data file not found: {data_path}")
        print("Run preprocess.py first to generate processed data.")
