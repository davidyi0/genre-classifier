import numpy as np
from typing import Dict, List, Tuple, Any
import librosa

def normalize_features(features: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize features using different methods.
    
    Args:
        features: Input features array
        method: Normalization method ('minmax', 'zscore', 'robust')
    
    Returns:
        Normalized features
    """
    if method == 'minmax':
        min_val = np.min(features)
        max_val = np.max(features)
        if max_val - min_val > 0:
            return (features - min_val) / (max_val - min_val)
        return features
    elif method == 'zscore':
        mean_val = np.mean(features)
        std_val = np.std(features)
        if std_val > 0:
            return (features - mean_val) / std_val
        return features
    elif method == 'robust':
        median_val = np.median(features)
        mad = np.median(np.abs(features - median_val))
        if mad > 0:
            return (features - median_val) / mad
        return features
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def pad_or_truncate(features: np.ndarray, target_length: int, pad_value: float = 0.0) -> np.ndarray:
    """
    Pad or truncate features to target length.
    
    Args:
        features: Input features array
        target_length: Desired length
        pad_value: Value to use for padding
    
    Returns:
        Padded/truncated features
    """
    current_length = features.shape[-1]
    
    if current_length < target_length:
        # Pad with zeros
        pad_width = target_length - current_length
        if len(features.shape) == 1:
            return np.pad(features, (0, pad_width), mode='constant', constant_values=pad_value)
        else:
            return np.pad(features, ((0, 0), (0, pad_width)), mode='constant', constant_values=pad_value)
    elif current_length > target_length:
        # Truncate
        if len(features.shape) == 1:
            return features[:target_length]
        else:
            return features[:, :target_length]
    else:
        return features

def prepare_cnn_features(features_dict: Dict[str, Any], target_length: int = 1000) -> Dict[str, np.ndarray]:
    """
    Prepare features for 1D CNN input.
    
    Args:
        features_dict: Dictionary containing extracted features
        target_length: Target length for time series features
    
    Returns:
        Dictionary with prepared CNN features
    """
    cnn_features = {}
    
    # Process time-series features
    time_series_features = [
        'mfccs', 'mel_spectrogram', 'chroma', 'spectral_centroids',
        'spectral_rolloff', 'spectral_bandwidth', 'zero_crossing_rate',
        'rms_energy', 'spectral_contrast', 'tonnetz'
    ]
    
    for feature_name in time_series_features:
        if feature_name in features_dict:
            feature_data = np.array(features_dict[feature_name])
            
            # Normalize features
            if len(feature_data.shape) == 2:
                # 2D features (e.g., MFCCs, mel spectrogram)
                normalized = np.array([normalize_features(feature_data[i, :]) for i in range(feature_data.shape[0])])
                padded = pad_or_truncate(normalized, target_length)
            else:
                # 1D features
                normalized = normalize_features(feature_data)
                padded = pad_or_truncate(normalized, target_length)
            
            cnn_features[f"{feature_name}_cnn"] = padded
    
    # Create combined feature vector
    combined_features = []
    for feature_name in time_series_features:
        if f"{feature_name}_cnn" in cnn_features:
            combined_features.append(cnn_features[f"{feature_name}_cnn"])
    
    if combined_features:
        cnn_features['combined_features'] = np.vstack(combined_features)
    
    return cnn_features

def extract_statistical_features(features_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract statistical features from time-series data.
    
    Args:
        features_dict: Dictionary containing extracted features
    
    Returns:
        Dictionary with statistical features
    """
    stats_features = {}
    
    time_series_features = [
        'mfccs', 'mel_spectrogram', 'chroma', 'spectral_centroids',
        'spectral_rolloff', 'spectral_bandwidth', 'zero_crossing_rate',
        'rms_energy', 'spectral_contrast', 'tonnetz'
    ]
    
    for feature_name in time_series_features:
        if feature_name in features_dict:
            feature_data = np.array(features_dict[feature_name])
            
            if len(feature_data.shape) == 2:
                # 2D features - compute stats across time dimension
                stats_features[f"{feature_name}_mean"] = float(np.mean(feature_data))
                stats_features[f"{feature_name}_std"] = float(np.std(feature_data))
                stats_features[f"{feature_name}_min"] = float(np.min(feature_data))
                stats_features[f"{feature_name}_max"] = float(np.max(feature_data))
                stats_features[f"{feature_name}_median"] = float(np.median(feature_data))
            else:
                # 1D features
                stats_features[f"{feature_name}_mean"] = float(np.mean(feature_data))
                stats_features[f"{feature_name}_std"] = float(np.std(feature_data))
                stats_features[f"{feature_name}_min"] = float(np.min(feature_data))
                stats_features[f"{feature_name}_max"] = float(np.max(feature_data))
                stats_features[f"{feature_name}_median"] = float(np.median(feature_data))
    
    return stats_features

def get_feature_summary(features_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary of extracted features for display.
    
    Args:
        features_dict: Dictionary containing extracted features
    
    Returns:
        Summary dictionary
    """
    summary = {
        "basic_info": {
            "duration": features_dict.get("duration", 0),
            "tempo": features_dict.get("tempo", 0),
            "sample_rate": features_dict.get("sample_rate", 0),
            "audio_length": features_dict.get("audio_length", 0)
        },
        "feature_shapes": features_dict.get("feature_shapes", {}),
        "statistical_features": extract_statistical_features(features_dict)
    }
    
    return summary 