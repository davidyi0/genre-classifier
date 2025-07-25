# Music Genre Identifier

A full-stack web application that analyzes audio files and predicts their music genre using machine learning. The application features a modern React frontend with a FastAPI backend that leverages librosa for audio feature extraction and a custom CNN model for genre classification.

## Features

- **Audio Analysis**: Extract comprehensive audio features including MFCCs, mel-spectrograms, chroma features, and more
- **Genre Prediction**: Predict music genre using a trained 1D CNN model
- **Real-time Processing**: Upload MP3 files and get instant analysis results
- **Modern UI**: Clean, responsive interface built with React and Tailwind CSS
- **RESTful API**: FastAPI backend with CORS support for seamless frontend-backend communication

## Architecture

### Frontend (React + TypeScript)
- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS for modern, responsive design
- **Icons**: Lucide React for clean iconography
- **Build Tool**: Vite for fast development and building

### Backend (Python + FastAPI)
- **Framework**: FastAPI for high-performance API
- **Audio Processing**: librosa for audio feature extraction
- **Machine Learning**: PyTorch for CNN model training and inference
- **Data Processing**: NumPy, Pandas, and scikit-learn

## Machine Learning Model

### Model Architecture
- **Type**: 1D Convolutional Neural Network (CNN)
- **Input**: Audio features (MFCCs, mel-spectrograms, chroma, etc.)
- **Output**: Genre classification with confidence scores
- **Training Data**: FMA (Free Music Archive) dataset

### Features Extracted
- **MFCCs** (Mel-frequency cepstral coefficients)
- **Mel-spectrograms** (Mel-frequency spectrograms)
- **Chroma features** (Harmonic content)
- **Spectral features** (Centroid, rolloff, bandwidth)
- **Rhythm features** (Tempo, beat tracking)
- **Harmonic features** (Tonnetz, spectral contrast)

## Project Structure

```
Music Genre Identifier/
├── backend/                 # Python FastAPI backend
│   ├── main.py             # FastAPI application with endpoints
│   ├── model_data.py       # CNN model and dataset processing
│   ├── feature_utils.py    # Audio feature extraction utilities
│   ├── requirements.txt    # Python dependencies
│   ├── models/             # Trained model files
│   └── fma_dataset/        # FMA dataset (music files and metadata)
├── src/                    # React frontend source
│   ├── App.tsx            # Main application component
│   ├── main.tsx           # React entry point
│   └── index.css          # Global styles
├── package.json           # Node.js dependencies and scripts
└── README.md             
```

## Quick Start


**Note**: This application requires the FMA dataset to be present in the `backend/fma_dataset/` directory for full functionality. The dataset is not included in this repository due to size constraints. 

### Prerequisites

- **Node.js** (v16 or higher)
- **Python** (3.8 or higher)
- **pip** (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Music-Genre-Identifier
   ```

2. **Set up the frontend**
   ```bash
   npm install
   ```

3. **Set up the backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the backend server**
   ```bash
   cd backend
   python main.py
   ```
   The API will be available at `http://localhost:8000`

2. **Start the frontend development server**
   ```bash
   npm run dev
   ```
   The application will be available at `http://localhost:5173`

3. **Open your browser** and navigate to `http://localhost:5173`

## Usage

### Web Interface

1. **Upload Audio**: Drag and drop an MP3 file or click to browse
2. **Choose Analysis Mode**:
   - **Feature Extraction**: Get detailed audio features and statistics
   - **Genre Prediction**: Predict the music genre with confidence score
3. **View Results**: See analysis results including duration, tempo, key, loudness, and spectral features

### API Endpoints

The backend provides several REST endpoints:

- `POST /extract-features` - Extract basic audio features
- `POST /extract-cnn-features` - Extract features optimized for CNN input
- `POST /predict-genre` - Predict genre using trained model
- `GET /health` - Health check endpoint

## Development

### Frontend Development
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run lint         # Run ESLint
npm run preview      # Preview production build
```

### Backend Development
```bash
cd backend
python main.py       # Run FastAPI server
```

### Model Training
The CNN model can be retrained using the FMA dataset:
```bash
cd backend
python model_data.py
```

## Dataset

This project uses the **FMA (Free Music Archive)** dataset for training the genre classification model. The dataset includes:
- Audio files in MP3 format
- Metadata with genre labels
- Multiple subsets (small, medium, large)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Acknowledgments

- **FMA Dataset**: Free Music Archive for providing the training dataset
- **librosa**: For excellent audio processing capabilities
- **FastAPI**: For the high-performance web framework
- **React & Vite**: For the modern frontend development experience

