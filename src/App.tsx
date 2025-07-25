import React, { useState, useRef } from 'react';
import { Upload, Music, BarChart3, FileAudio, CheckCircle } from 'lucide-react';

interface AnalysisResult {
  duration: number;
  tempo: number;
  key: string;
  loudness: number;
  spectralCentroid: number;
}

interface CNNFeatures {
  cnn_features: {
    [key: string]: number[][];
  };
  feature_summary: {
    basic_info: {
      duration: number;
      tempo: number;
      sample_rate: number;
      audio_length: number;
    };
    feature_shapes: {
      [key: string]: number[];
    };
    statistical_features: {
      [key: string]: number;
    };
  };
  target_length: number;
}

interface GenrePrediction {
  genre: string;
  confidence: number;
  features_shape: number[];
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [cnnFeatures, setCnnFeatures] = useState<CNNFeatures | null>(null);
  const [genrePrediction, setGenrePrediction] = useState<GenrePrediction | null>(null);
  const [useRealAnalysis, setUseRealAnalysis] = useState(true);
  const [analysisMode, setAnalysisMode] = useState<'features' | 'genre'>('features');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (selectedFile: File) => {
    if (selectedFile.type === 'audio/mpeg' || selectedFile.name.endsWith('.mp3')) {
      setFile(selectedFile);
      setAnalysisResult(null);
    } else {
      alert('Please select a valid MP3 file');
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      handleFileSelect(droppedFile);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      handleFileSelect(selectedFile);
    }
  };

  const analyzeFile = async () => {
    if (!file) return;
    
    setIsAnalyzing(true);
    
    try {
      if (useRealAnalysis) {
        const formData = new FormData();
        formData.append('file', file);
        
        if (analysisMode === 'genre') {
          // Predict genre
          const response = await fetch('http://localhost:8000/predict-genre', {
            method: 'POST',
            body: formData,
          });
          
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          
          const result = await response.json();
          setGenrePrediction(result);
        } else {
          // Extract features
          const response = await fetch('http://localhost:8000/extract-cnn-features', {
            method: 'POST',
            body: formData,
          });
          
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          
          const result = await response.json();
          setCnnFeatures(result);
          
          // Create a simplified analysis result for display
          const basicInfo = result.feature_summary.basic_info;
          const stats = result.feature_summary.statistical_features;
          
          const analysisResult: AnalysisResult = {
            duration: basicInfo.duration,
            tempo: basicInfo.tempo,
            key: 'C', // need to implement key detection
            loudness: stats.rms_energy_mean || -10,
            spectralCentroid: stats.spectral_centroids_mean || 1500
          };
          
          setAnalysisResult(analysisResult);
        }
      } else {
        // mockdata for testing
    setTimeout(() => {
      const mockResult: AnalysisResult = {
        duration: Math.random() * 240 + 60, // 1-5 minutes
        tempo: Math.floor(Math.random() * 60 + 80), // 80-140 BPM
        key: ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][Math.floor(Math.random() * 12)],
        loudness: Math.random() * -10 - 5, // -15 to -5 dB
        spectralCentroid: Math.random() * 2000 + 1000 // 1000-3000 Hz
      };
      
      setAnalysisResult(mockResult);
      setIsAnalyzing(false);
    }, 2000);
      }
    } catch (error) {
      console.error('Error analyzing file:', error);
      alert('Error analyzing file. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const resetApp = () => {
    setFile(null);
    setAnalysisResult(null);
    setCnnFeatures(null);
    setGenrePrediction(null);
    setIsAnalyzing(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-2xl mx-auto">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Music className="w-8 h-8 text-blue-600 mr-2" />
            <h1 className="text-3xl font-bold text-gray-900">Audio Analyzer</h1>
          </div>
          <p className="text-gray-600">Upload MP3 files for detailed audio analysis with librosa feature extraction</p>
          
          <div className="mt-4 flex items-center justify-center space-x-6">
            <label className="flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={useRealAnalysis}
                onChange={(e) => setUseRealAnalysis(e.target.checked)}
                className="sr-only"
              />
              <div className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                useRealAnalysis ? 'bg-blue-600' : 'bg-gray-300'
              }`}>
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  useRealAnalysis ? 'translate-x-6' : 'translate-x-1'
                }`} />
              </div>
              <span className="ml-3 text-sm text-gray-700">
                {useRealAnalysis ? 'Real Analysis (Backend)' : 'Mock Analysis'}
              </span>
            </label>
            
            {useRealAnalysis && (
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-700">Mode:</span>
                <select
                  value={analysisMode}
                  onChange={(e) => setAnalysisMode(e.target.value as 'features' | 'genre')}
                  className="text-sm border border-gray-300 rounded px-2 py-1"
                >
                  <option value="features">Feature Extraction</option>
                  <option value="genre">Genre Prediction</option>
                </select>
              </div>
            )}
          </div>
        </div>

        {!file ? (
          <div
            className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-blue-400 transition-colors cursor-pointer bg-white"
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-xl text-gray-600 mb-2">Drop your MP3 file here</p>
            <p className="text-gray-500">or click to browse</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp3,audio/mpeg"
              onChange={handleFileInputChange}
              className="hidden"
            />
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center">
                <FileAudio className="w-8 h-8 text-blue-600 mr-3" />
                <div>
                  <h3 className="font-semibold text-gray-900">{file.name}</h3>
                  <p className="text-sm text-gray-500">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
                </div>
              </div>
              <button
                onClick={resetApp}
                className="text-gray-500 hover:text-gray-700 text-sm font-medium"
              >
                Remove
              </button>
            </div>

            {!analysisResult && !isAnalyzing && (
              <button
                onClick={analyzeFile}
                className="w-full bg-blue-600 text-white py-3 px-4 rounded-md hover:bg-blue-700 transition-colors font-medium flex items-center justify-center"
              >
                <BarChart3 className="w-4 h-4 mr-2" />
                Analyze Audio
              </button>
            )}

            {isAnalyzing && (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <p className="text-gray-600">Analyzing audio file...</p>
              </div>
            )}

            {analysisResult && (
              <div className="border-t pt-6">
                <div className="flex items-center mb-4">
                  <CheckCircle className="w-5 h-5 text-green-600 mr-2" />
                  <h3 className="font-semibold text-gray-900">Analysis Complete</h3>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 rounded-md p-4">
                    <p className="text-sm text-gray-600 mb-1">Duration</p>
                    <p className="text-lg font-semibold text-gray-900">{formatDuration(analysisResult.duration)}</p>
                  </div>
                  
                  <div className="bg-gray-50 rounded-md p-4">
                    <p className="text-sm text-gray-600 mb-1">Tempo</p>
                    <p className="text-lg font-semibold text-gray-900">{analysisResult.tempo} BPM</p>
                  </div>
                  
                  <div className="bg-gray-50 rounded-md p-4">
                    <p className="text-sm text-gray-600 mb-1">Key</p>
                    <p className="text-lg font-semibold text-gray-900">{analysisResult.key}</p>
                  </div>
                  
                  <div className="bg-gray-50 rounded-md p-4">
                    <p className="text-sm text-gray-600 mb-1">Loudness</p>
                    <p className="text-lg font-semibold text-gray-900">{analysisResult.loudness.toFixed(1)} dB</p>
                  </div>
                  
                  <div className="bg-gray-50 rounded-md p-4 col-span-2">
                    <p className="text-sm text-gray-600 mb-1">Spectral Centroid</p>
                    <p className="text-lg font-semibold text-gray-900">{analysisResult.spectralCentroid.toFixed(0)} Hz</p>
                  </div>
                </div>
                
                <button
                  onClick={resetApp}
                  className="w-full mt-6 bg-gray-100 text-gray-700 py-2 px-4 rounded-md hover:bg-gray-200 transition-colors font-medium"
                >
                  Analyze Another File
                </button>
              </div>
            )}
            
            {/* CNN Features Display */}
            {cnnFeatures && (
              <div className="mt-8 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">CNN-Ready Features</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium text-gray-700 mb-2">Feature Summary</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Target Length:</span>
                        <span className="font-mono">{cnnFeatures.target_length}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Duration:</span>
                        <span className="font-mono">{cnnFeatures.feature_summary.basic_info.duration.toFixed(2)}s</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Sample Rate:</span>
                        <span className="font-mono">{cnnFeatures.feature_summary.basic_info.sample_rate} Hz</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Audio Length:</span>
                        <span className="font-mono">{cnnFeatures.feature_summary.basic_info.audio_length.toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium text-gray-700 mb-2">Available Features</h4>
                    <div className="space-y-1 text-sm">
                      {Object.keys(cnnFeatures.cnn_features).map((featureName) => (
                        <div key={featureName} className="flex justify-between">
                          <span className="capitalize">{featureName.replace('_cnn', '')}:</span>
                          <span className="font-mono text-gray-600">
                            {cnnFeatures.cnn_features[featureName].length} × {cnnFeatures.cnn_features[featureName][0]?.length || 0}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                
                <div className="mt-6">
                  <h4 className="font-medium text-gray-700 mb-2">Feature Shapes</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    {Object.entries(cnnFeatures.feature_summary.feature_shapes).map(([name, shape]) => (
                      <div key={name} className="bg-gray-50 rounded p-2">
                        <div className="font-medium text-gray-600">{name}</div>
                        <div className="font-mono text-xs">[{shape.join(', ')}]</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
            
            {/* Genre Prediction Display */}
            {genrePrediction && (
              <div className="mt-8 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Genre Prediction</h3>
                
                <div className="text-center">
                  <div className="bg-blue-50 rounded-lg p-6 mb-4">
                    <h4 className="text-2xl font-bold text-blue-900 mb-2">
                      {genrePrediction.genre}
                    </h4>
                    <p className="text-blue-700">
                      Confidence: {(genrePrediction.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                  
                  <div className="text-sm text-gray-600">
                    <p>Features shape: [{genrePrediction.features_shape.join(', ')}]</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;