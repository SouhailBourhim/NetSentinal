"""
═══════════════════════════════════════════════════════════════
NetSentinel — Feature Engineer
Creates new features from network traffic data
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from pathlib import Path


class NetworkFeatureEngineer:
    """
    Generates derived features from network traffic data
    to improve ML model performance.
    """

    def __init__(self):
        self.engineered_features = []

    def _get_column_mapping(self, df: pd.DataFrame) -> dict:
        """
        Create a mapping of expected column names to actual column names.
        This handles variations in column naming conventions.
        """
        columns = df.columns.tolist()
        
        # Create a mapping of standardized names to actual column names
        mapping = {}
        
        for col in columns:
            col_lower = col.lower().strip()
            
            # Map common variations
            if 'total fwd packets' in col_lower:
                mapping['total_fwd_packets'] = col
            elif 'total backward packets' in col_lower:
                mapping['total_backward_packets'] = col
            elif 'total length of fwd packets' in col_lower:
                mapping['total_length_of_fwd_packets'] = col
            elif 'total length of bwd packets' in col_lower:
                mapping['total_length_of_bwd_packets'] = col
            elif 'flow duration' in col_lower:
                mapping['flow_duration'] = col
            elif 'fwd packet length mean' in col_lower:
                mapping['fwd_packet_length_mean'] = col
            elif 'fwd packet length std' in col_lower:
                mapping['fwd_packet_length_std'] = col
            elif 'flow iat mean' in col_lower:
                mapping['flow_iat_mean'] = col
            elif 'flow iat std' in col_lower:
                mapping['flow_iat_std'] = col
                
        return mapping

    def create_packet_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratios between forward and backward traffic.
        Attacks often have asymmetric traffic patterns.

        Example:
        - DDoS: many forward packets, few backward
        - Port scan: many short forward flows, almost no response
        - Normal browsing: balanced forward/backward
        """
        print("  → Creating packet ratios...")

        # Use actual column names from the dataset
        fwd_packets_col = 'Total Fwd Packets'
        bwd_packets_col = 'Total Backward Packets'
        fwd_bytes_col = 'Total Length of Fwd Packets'
        bwd_bytes_col = 'Total Length of Bwd Packets'
        duration_col = 'Flow Duration'

        # Ratio of forward to total packets
        if fwd_packets_col in df.columns and bwd_packets_col in df.columns:
            total_packets = df[fwd_packets_col] + df[bwd_packets_col]
            df['fwd_packet_ratio'] = np.where(
                total_packets > 0,
                df[fwd_packets_col] / total_packets,
                0
            )
            self.engineered_features.append('fwd_packet_ratio')

        # Ratio of forward to backward bytes
        if fwd_bytes_col in df.columns and bwd_bytes_col in df.columns:
            total_bytes = df[fwd_bytes_col] + df[bwd_bytes_col]
            df['fwd_bytes_ratio'] = np.where(
                total_bytes > 0,
                df[fwd_bytes_col] / total_bytes,
                0
            )
            self.engineered_features.append('fwd_bytes_ratio')

        # Packets per second (if not already present)
        if duration_col in df.columns and fwd_packets_col in df.columns and bwd_packets_col in df.columns:
            duration_seconds = df[duration_col] / 1e6  # microseconds to seconds
            total_packets = df[fwd_packets_col] + df[bwd_packets_col]
            df['packets_per_second'] = np.where(
                duration_seconds > 0,
                total_packets / duration_seconds,
                0
            )
            self.engineered_features.append('packets_per_second')

        return df

    def create_flag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from TCP flag counts.
        TCP flags reveal connection behavior:

        - High SYN without ACK = SYN flood attack
        - High RST = connection resets (suspicious)
        - High PSH = data pushing (could be exfiltration)
        """
        print("  → Creating flag features...")

        flag_columns = [col for col in df.columns if 'flag' in col.lower()]

        if len(flag_columns) >= 2:
            # Total flags
            df['total_flags'] = df[flag_columns].sum(axis=1)

            # SYN to ACK ratio (if columns exist)
            syn_col = [col for col in flag_columns if 'syn' in col.lower()]
            ack_col = [col for col in flag_columns if 'ack' in col.lower()]

            if syn_col and ack_col:
                df['syn_ack_ratio'] = np.where(
                    df[ack_col[0]] > 0,
                    df[syn_col[0]] / df[ack_col[0]],
                    df[syn_col[0]]
                )
                self.engineered_features.append('syn_ack_ratio')

            self.engineered_features.append('total_flags')
        return df

    def create_flow_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create flow intensity features.
        Intensity = how much data is packed into a flow.

        High intensity in short duration = likely attack.
        Low intensity over long duration = likely normal.
        """
        print("  → Creating flow intensity features...")

        duration_col = 'Flow Duration'
        fwd_packets_col = 'Total Fwd Packets'
        bwd_packets_col = 'Total Backward Packets'
        fwd_bytes_col = 'Total Length of Fwd Packets'
        bwd_bytes_col = 'Total Length of Bwd Packets'

        if duration_col in df.columns:
            duration_seconds = df[duration_col] / 1e6
            duration_safe = duration_seconds.replace(0, 1e-6)

            # Bytes per second
            if fwd_bytes_col in df.columns and bwd_bytes_col in df.columns:
                total_bytes = df[fwd_bytes_col] + df[bwd_bytes_col]
                df['bytes_per_second'] = total_bytes / duration_safe
                self.engineered_features.append('bytes_per_second')

            # Packets per flow duration
            if fwd_packets_col in df.columns and bwd_packets_col in df.columns:
                total_packets = df[fwd_packets_col] + df[bwd_packets_col]
                df['packet_density'] = total_packets / duration_safe
                self.engineered_features.append('packet_density')

        return df

    def create_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Shannon entropy-based features.

        Shannon entropy measures randomness/diversity.
        - High entropy in destination ports = port scanning
        - Low entropy = normal traffic (few ports used)

        This is one of the MOST powerful features for
        network anomaly detection.
        """
        print("  → Creating entropy features...")

        # Packet size entropy (using forward packet length stats)
        fwd_mean_col = 'Fwd Packet Length Mean'
        fwd_std_col = 'Fwd Packet Length Std'

        if fwd_mean_col in df.columns and fwd_std_col in df.columns:
            # Coefficient of variation as proxy for entropy
            df['packet_size_variation'] = np.where(
                df[fwd_mean_col] > 0,
                df[fwd_std_col] / df[fwd_mean_col],
                0
            )
            self.engineered_features.append('packet_size_variation')

        # IAT (Inter-Arrival Time) variation
        iat_mean_col = 'Flow IAT Mean'
        iat_std_col = 'Flow IAT Std'

        if iat_mean_col in df.columns and iat_std_col in df.columns:
            df['iat_variation'] = np.where(
                df[iat_mean_col] > 0,
                df[iat_std_col] / df[iat_mean_col],
                0
            )
            self.engineered_features.append('iat_variation')

        return df

    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral pattern features.
        These capture HOW a connection behaves, not just WHAT it sends.

        - Is the flow bidirectional or one-way?
        - Is the traffic bursty or steady?
        - Are packets uniformly sized or variable?
        """
        print("  → Creating behavioral features...")

        fwd_packets_col = 'Total Fwd Packets'
        bwd_packets_col = 'Total Backward Packets'
        fwd_mean_col = 'Fwd Packet Length Mean'
        duration_col = 'Flow Duration'

        # Bidirectionality score
        if fwd_packets_col in df.columns and bwd_packets_col in df.columns:
            fwd = df[fwd_packets_col]
            bwd = df[bwd_packets_col]
            total = fwd + bwd

            df['bidirectional_score'] = np.where(
                total > 0,
                2 * np.minimum(fwd, bwd) / total,
                0
            )
            # 0 = completely one-directional
            # 1 = perfectly balanced
            self.engineered_features.append('bidirectional_score')

        # Small packet indicator
        if fwd_mean_col in df.columns:
            df['is_small_packet_flow'] = (df[fwd_mean_col] < 100).astype(int)
            self.engineered_features.append('is_small_packet_flow')

        # Short flow indicator
        if duration_col in df.columns:
            df['is_short_flow'] = (df[duration_col] < 1000).astype(int)
            self.engineered_features.append('is_short_flow')

        return df

    def handle_infinites(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace any infinite values created during feature engineering.
        """
        print("  → Cleaning infinite values from new features...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(df[numeric_cols]).sum().sum()

        if inf_count > 0:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)
            print(f"    Replaced {inf_count} infinite values")
        else:
            print("    ✅ No infinite values found")

        return df

    def run_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete feature engineering pipeline.
        """
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)

        cols_before = len(df.columns)
        
        # Get column mapping for robust column name handling
        self.column_mapping = self._get_column_mapping(df)
        print(f"  Found {len(self.column_mapping)} mapped columns")

        df = self.create_packet_ratios(df)
        df = self.create_flag_features(df)
        df = self.create_flow_intensity(df)
        df = self.create_entropy_features(df)
        df = self.create_behavioral_features(df)
        df = self.handle_infinites(df)

        cols_after = len(df.columns)

        print(f"\n  ✅ Feature engineering complete")
        print(f"     Columns before: {cols_before}")
        print(f"     Columns after:  {cols_after}")
        print(f"     New features:   {cols_after - cols_before}")
        print(f"\n  New features created:")
        for feat in self.engineered_features:
            if feat in df.columns:
                print(f"    → {feat}")

        return df