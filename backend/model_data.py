import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FMADatasetProcessor:
    def __init__(self, fma_root_path, subset='small'):
        """
        Initialize FMA dataset processor
        
        Args:
            fma_root_path: Path to FMA dataset root directory
            subset: 'small', 'medium', or 'large' - determines which subset to use
        """
        self.fma_root_path = Path(fma_root_path)
        self.subset = subset
        self.tracks_df = None
        self.genres_df = None
        self.features_df = None
        self.label_encoder = LabelEncoder()
        
    def load_metadata(self):
        """Load FMA metadata files"""
        print("Loading FMA metadata...")
        
        # Load tracks metadata
        tracks_path = self.fma_root_path / 'fma_metadata' / 'tracks.csv'
        if not tracks_path.exists():
            raise FileNotFoundError(f"Tracks metadata not found at {tracks_path}")
        self.tracks_df = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
        
        # Load genres metadata
        genres_path = self.fma_root_path / 'fma_metadata' / 'genres.csv'
        if not genres_path.exists():
            raise FileNotFoundError(f"Genres metadata not found at {genres_path}")
        self.genres_df = pd.read_csv(genres_path, index_col=0)
        
        # Load features if available??
        features_path = self.fma_root_path / 'fma_metadata' / 'features.csv'
        if features_path.exists():
            self.features_df = pd.read_csv(features_path, index_col=0, header=[0, 1, 2])
            
        print(f"Loaded metadata for {len(self.tracks_df)} tracks")
        
    def get_track_subset(self):
        """Get track IDs for the specified subset"""
        if self.subset == 'small':
            subset_tracks = self.tracks_df[self.tracks_df['set', 'subset'] <= 'small'].index
        elif self.subset == 'medium':
            subset_tracks = self.tracks_df[self.tracks_df['set', 'subset'] <= 'medium'].index
        else:
            subset_tracks = self.tracks_df.index
            
        return subset_tracks
    
    def get_genre_labels(self, track_ids, min_samples_per_class=5):
        """Extract genre labels for given track IDs, filtering out rare classes"""
        labels = []
        valid_track_ids = []
        
        # first iteration: collect all labels
        for track_id in track_ids:
            try:
                # Get the top genre for this track
                genre_top = self.tracks_df.loc[track_id, ('track', 'genre_top')]
                if pd.notna(genre_top):
                    labels.append(genre_top)
                    valid_track_ids.append(track_id)
            except KeyError:
                continue
        
        # Count genre frequencies
        from collections import Counter
        genre_counts = Counter(labels)
        
        # Filter out genres with too few samples
        common_genres = {genre for genre, count in genre_counts.items() if count >= min_samples_per_class}
        
        print(f"Genre distribution before filtering:")
        for genre, count in sorted(genre_counts.items()):
            print(f"  {genre}: {count} samples")
        
        print(f"\nKeeping genres with at least {min_samples_per_class} samples:")
        for genre in sorted(common_genres):
            print(f"  {genre}: {genre_counts[genre]} samples")
        
        # Second pass: keep only common genres
        filtered_labels = []
        filtered_track_ids = []
        
        for track_id, label in zip(valid_track_ids, labels):
            if label in common_genres:
                filtered_labels.append(label)
                filtered_track_ids.append(track_id)
        
        print(f"\nFinal dataset: {len(filtered_labels)} tracks from {len(common_genres)} genres")
        
        return filtered_track_ids, filtered_labels
    
    def extract_time_series_features(self, audio_path, sr=22050, n_mels=128, hop_length=512):
        """
        Extract time-series features from audio file
        This should match your existing feature extraction logic
        """
        try:

            y, sr_loaded = librosa.load(audio_path, sr=int(sr))
            
            duration = librosa.get_duration(y=y, sr=sr)
            
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            rms = librosa.feature.rms(y=y)[0]
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            features = np.vstack([
                mel_spec_db,      
                mfccs,            
                chroma,           
                spectral_centroids.reshape(1, -1), 
                spectral_rolloff.reshape(1, -1),   
                spectral_bandwidth.reshape(1, -1), 
                zero_crossing_rate.reshape(1, -1), 
                rms.reshape(1, -1),             
                spectral_contrast,               
                tonnetz                          
            ])
            
            return features
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def prepare_cnn_features(self, features, target_length=1292):
        """
        Convert variable-length features to fixed-length for CNN
        """
        if features is None:
            return None
            
        # If features are shorter than target, pad with zeros
        if features.shape[1] < target_length:
            padding = target_length - features.shape[1]
            features = np.pad(features, ((0, 0), (0, padding)), mode='constant')
        
        # If features are longer than target, truncate
        elif features.shape[1] > target_length:
            features = features[:, :target_length]
            
        return features
    
    def get_audio_path(self, track_id):
        """Get audio file path for a given track ID"""
        # FMA audio files are organized in subdirectories: fma_small/000/000001.mp3
        track_str = f"{track_id:06d}"
        subdir = track_str[:3]
        filename = f"{track_str}.mp3"
        
        if self.subset == 'small':
            audio_path = self.fma_root_path / 'fma_small' / subdir / filename
        elif self.subset == 'medium':
            audio_path = self.fma_root_path / 'fma_medium' / subdir / filename
        else:
            audio_path = self.fma_root_path / 'fma_large' / subdir / filename
            
        return audio_path
    
    def process_dataset(self, output_dir='processed_data', max_samples=None):
        """
        Process the entire FMA dataset and save features and labels
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        self.load_metadata()
        
        track_ids = self.get_track_subset()
        
        valid_track_ids, genre_labels = self.get_genre_labels(track_ids)
        
        if max_samples:
            valid_track_ids = valid_track_ids[:max_samples]
            genre_labels = genre_labels[:max_samples]
        
        print(f"Processing {len(valid_track_ids)} tracks...")
        
        all_features = []
        all_labels = []
        processed_count = 0
        
        for i, (track_id, label) in enumerate(zip(valid_track_ids, genre_labels)):
            if i % 100 == 0:
                print(f"Processed {i}/{len(valid_track_ids)} tracks...")
                
            audio_path = self.get_audio_path(track_id)
            
            if not audio_path.exists():
                continue
                
            features = self.extract_time_series_features(audio_path)
            if features is None:
                continue
                
            # Prepare features for CNN
            cnn_features = self.prepare_cnn_features(features)
            if cnn_features is None:
                continue
                
            all_features.append(cnn_features)
            all_labels.append(label)
            processed_count += 1
            
        print(f"Successfully processed {processed_count} tracks")
        
        if processed_count == 0:
            raise ValueError("No tracks were successfully processed")
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Save processed data
        np.save(output_dir / 'features.npy', X)
        np.save(output_dir / 'labels.npy', y_encoded)
        np.save(output_dir / 'label_names.npy', self.label_encoder.classes_)
        
        with open(output_dir / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
        print(f"Saved features shape: {X.shape}")
        print(f"Saved labels shape: {y_encoded.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        return X, y_encoded, self.label_encoder.classes_


class AudioDataset(Dataset):
    """PyTorch Dataset for audio features"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class AudioCNN(nn.Module):
    """CNN model for audio genre classification"""
    
    def __init__(self, input_shape, num_classes):
        super(AudioCNN, self).__init__()
        
        # Input shape: (batch_size, n_features, time_steps)
        n_features, time_steps = input_shape
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            
            # Second conv block
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            
            # Third conv block
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            
            # Fourth conv block
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Apply conv layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        
        return x


class AudioGenreClassifier:
    """Complete audio genre classification system"""
    
    def __init__(self, model_path='models'):
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.model = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train_model(self, features_path, labels_path, label_encoder_path):
        """Train the CNN model"""
        print("Loading processed data...")
        
        X = np.load(features_path)
        y = np.load(labels_path)
        
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
            
        num_classes = len(self.label_encoder.classes_)
        input_shape = X.shape[1:]  # (n_features, time_steps)
        
        print(f"Input shape: {input_shape}")
        print(f"Number of classes: {num_classes}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create datasets
        train_dataset = AudioDataset(X_train, y_train)
        test_dataset = AudioDataset(X_test, y_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        self.model = AudioCNN(input_shape, num_classes).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # Training loop
        print("Starting training...")
        num_epochs = 50
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            scheduler.step()
            
            # Validation
            self.model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_features)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += batch_labels.size(0)
                    test_correct += (predicted == batch_labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            test_acc = 100 * test_correct / test_total
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Acc: {test_acc:.2f}%')
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                self.save_model('best_model.pth')
                
        print(f"Training completed. Best accuracy: {best_acc:.2f}%")
        
    def save_model(self, filename):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'model_params': {
                'input_shape': self.model.conv_layers[0].in_channels,
                'num_classes': len(self.label_encoder.classes_)
            }
        }, self.model_path / filename)
        
    def load_model(self, filename):
        """Load a trained model"""
        checkpoint = torch.load(self.model_path / filename, map_location=self.device)
        
        # Reconstruct model
        input_shape = (checkpoint['model_params']['input_shape'], 1292)  # Assuming fixed length
        num_classes = checkpoint['model_params']['num_classes']
        
        self.model = AudioCNN(input_shape, num_classes).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.label_encoder = checkpoint['label_encoder']
        
    def predict_genre(self, audio_features):
        """Predict genre for given audio features"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        self.model.eval()
        
        # Prepare features
        features = torch.FloatTensor(audio_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
        genre = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return genre, confidence


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio Genre Classification with FMA Dataset')
    parser.add_argument('--fma_path', type=str, required=True, 
                       help='Path to FMA dataset root directory')
    parser.add_argument('--subset', type=str, default='small', 
                       choices=['small', 'medium', 'large'],
                       help='FMA subset to use')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                       help='Output directory for processed data')
    parser.add_argument('--train', action='store_true',
                       help='Train the model after processing data')
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = FMADatasetProcessor(
            fma_root_path=args.fma_path,
            subset=args.subset
        )
        
        # Process dataset
        print(f"Processing FMA {args.subset} subset...")
        X, y, class_names = processor.process_dataset(
            output_dir=args.output_dir,
            max_samples=args.max_samples
        )
        
        if args.train:
            # Initialize classifier
            classifier = AudioGenreClassifier()
            
            # Train model
            classifier.train_model(
                features_path=f'{args.output_dir}/features.npy',
                labels_path=f'{args.output_dir}/labels.npy',
                label_encoder_path=f'{args.output_dir}/label_encoder.pkl'
            )
            
            print("Training completed!")
        else:
            print("Data processing completed. Use --train flag to train the model.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the FMA dataset path is correct and the dataset is downloaded.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()