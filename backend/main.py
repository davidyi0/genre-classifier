from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import tempfile
import os
from typing import Dict, List, Any
import json
import torch
from feature_utils import prepare_cnn_features, get_feature_summary
from model_data import AudioGenreClassifier
# from self_supervised_learning import SelfSupervisedAudioEncoder

app = FastAPI(title="Audio Feature Extractor", version="1.0.0")

genre_classifier = None
self_supervised_model = None

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_time_series_features(audio_path: str, sr: int = 22050) -> Dict[str, Any]:
    """
    Extract time-series features suitable for 1D CNN using librosa.
    
    Args:
        audio_path: Path to the audio file
        sr: Sample rate for loading audio
    
    Returns:
        Dictionary containing extracted features
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
        
        features = {
            "duration": float(duration),
            "tempo": float(tempo),
            "sample_rate": sr,
            "audio_length": len(y),
            
            "mfccs": mfccs.tolist(),
            "mel_spectrogram": mel_spec_db.tolist(),
            "chroma": chroma.tolist(),
            "spectral_centroids": spectral_centroids.tolist(),
            "spectral_rolloff": spectral_rolloff.tolist(),
            "spectral_bandwidth": spectral_bandwidth.tolist(),
            "zero_crossing_rate": zero_crossing_rate.tolist(),
            "rms_energy": rms.tolist(),
            "spectral_contrast": spectral_contrast.tolist(),
            "tonnetz": tonnetz.tolist(),
            
            "mfccs_mean": np.mean(mfccs, axis=1).tolist(),
            "mfccs_std": np.std(mfccs, axis=1).tolist(),
            "chroma_mean": np.mean(chroma, axis=1).tolist(),
            "spectral_centroids_mean": float(np.mean(spectral_centroids)),
            "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
            "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
            "zero_crossing_rate_mean": float(np.mean(zero_crossing_rate)),
            "rms_energy_mean": float(np.mean(rms)),
            
            "feature_shapes": {
                "mfccs": mfccs.shape,
                "mel_spectrogram": mel_spec_db.shape,
                "chroma": chroma.shape,
                "spectral_centroids": spectral_centroids.shape,
                "spectral_rolloff": spectral_rolloff.shape,
                "spectral_bandwidth": spectral_bandwidth.shape,
                "zero_crossing_rate": zero_crossing_rate.shape,
                "rms_energy": rms.shape,
                "spectral_contrast": spectral_contrast.shape,
                "tonnetz": tonnetz.shape
            }
        }
        
        return features
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/extract-features")
async def extract_audio_features(file: UploadFile = File(...)):
    """
    Extract time-series features from uploaded audio file.
    """
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        try:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            features = extract_time_series_features(temp_file.name)
            
            return JSONResponse(content=features)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
        finally:
            temp_file.close()
            os.unlink(temp_file.name)

@app.post("/extract-cnn-features")
async def extract_cnn_features(file: UploadFile = File(...), target_length: int = 1000):
    """
    Extract and prepare features specifically for 1D CNN input.
    """
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        try:
            # Write uploaded file to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Extract raw features
            raw_features = extract_time_series_features(temp_file.name)
            
            # Prepare CNN features
            cnn_features = prepare_cnn_features(raw_features, target_length)
            
            # Convert numpy arrays to lists for JSON serialization
            cnn_features_serializable = {}
            for key, value in cnn_features.items():
                if isinstance(value, np.ndarray):
                    cnn_features_serializable[key] = value.tolist()
                else:
                    cnn_features_serializable[key] = value
            
            # Add feature summary
            feature_summary = get_feature_summary(raw_features)
            
            response = {
                "cnn_features": cnn_features_serializable,
                "feature_summary": feature_summary,
                "target_length": target_length
            }
            
            return JSONResponse(content=response)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
        finally:
            temp_file.close()
            os.unlink(temp_file.name)

@app.post("/predict-genre")
async def predict_genre(file: UploadFile = File(...)):
    """
    Predict genre for uploaded audio file using trained CNN model.
    """
    global genre_classifier
    
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Load model if not already loaded
    if genre_classifier is None:
        try:
            genre_classifier = AudioGenreClassifier()
            genre_classifier.load_model('best_model.pth')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        try:
            # Write uploaded file to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Extract features using the same method as training
            from model_data import FMADatasetProcessor
            processor = FMADatasetProcessor(fma_root_path=".")
            features = processor.extract_time_series_features(temp_file.name)
            
            if features is None:
                raise HTTPException(status_code=500, detail="Failed to extract features")
            
            # Prepare features for CNN (same as training)
            cnn_features = processor.prepare_cnn_features(features)
            
            if cnn_features is None:
                raise HTTPException(status_code=500, detail="Failed to prepare CNN features")
            
            # Predict genre
            genre, confidence = genre_classifier.predict_genre(cnn_features)
            
            return JSONResponse(content={
                "genre": genre,
                "confidence": confidence,
                "features_shape": cnn_features.shape
            })
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error predicting genre: {str(e)}")
        finally:
            # Clean up temporary file
            temp_file.close()
            os.unlink(temp_file.name)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Audio Feature Extractor is running"}

# @app.post("/predict-genre-self-supervised")
# async def predict_genre_self_supervised(file: UploadFile = File(...)):
#     """
#     Predict genre for uploaded audio file using self-supervised model.
#     """
#     global self_supervised_model
    
#     if not file.content_type or not file.content_type.startswith('audio/'):
#         raise HTTPException(status_code=400, detail="File must be an audio file")
    
#     # Load self-supervised model if not already loaded
#     if self_supervised_model is None:
#         try:
#             # Load the saved model
#             checkpoint = torch.load('self_supervised_model.pth', map_location='cpu')
            
#             # Get input dimension from the model
#             input_dim = 128  # This should match your feature dimension
#             num_classes = len(checkpoint['label_encoder'].classes_)
            
#             # Initialize encoder
#             encoder = SelfSupervisedAudioEncoder(
#                 input_dim=input_dim,
#                 hidden_dim=256,
#                 feature_dim=128
#             )
            
#             # Create full model with classifier
#             classifier = torch.nn.Linear(256, num_classes)  # 256 is hidden_dim
#             full_model = torch.nn.Sequential(encoder.encoder, classifier)
            
#             # Load state dict
#             full_model.load_state_dict(checkpoint['model_state_dict'])
#             full_model.eval()
            
#             self_supervised_model = {
#                 'model': full_model,
#                 'label_encoder': checkpoint['label_encoder']
#             }
            
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error loading self-supervised model: {str(e)}")
    
#     # Create temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
#         try:
#             # Write uploaded file to temporary file
#             content = await file.read()
#             temp_file.write(content)
#             temp_file.flush()
            
#             # Extract features using the same method as training
#             from model_data import FMADatasetProcessor
#             processor = FMADatasetProcessor(fma_root_path=".")
#             features = processor.extract_time_series_features(temp_file.name)
            
#             if features is None:
#                 raise HTTPException(status_code=500, detail="Failed to extract features")
            
#             # Prepare features for CNN (same as training)
#             cnn_features = processor.prepare_cnn_features(features)
            
#             if cnn_features is None:
#                 raise HTTPException(status_code=500, detail="Failed to prepare CNN features")
            
#             # Convert to tensor and predict
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             features_tensor = torch.FloatTensor(cnn_features).unsqueeze(0).to(device)
            
#             with torch.no_grad():
#                 outputs = self_supervised_model['model'](features_tensor)
#                 probabilities = torch.softmax(outputs, dim=1)
#                 confidence, predicted_idx = torch.max(probabilities, 1)
                
#                 # Get predicted genre
#                 predicted_genre = self_supervised_model['label_encoder'].classes_[predicted_idx.item()]
#                 confidence_value = confidence.item()
            
#             return JSONResponse(content={
#                 "genre": predicted_genre,
#                 "confidence": confidence_value,
#                 "model_type": "self_supervised",
#                 "features_shape": cnn_features.shape
#             })
            
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error predicting genre: {str(e)}")
#         finally:
#             # Clean up temporary file
#             temp_file.close()
#             os.unlink(temp_file.name)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Audio Feature Extractor API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 