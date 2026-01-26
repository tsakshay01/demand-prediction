import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { Upload as UploadIcon, FileText, CheckCircle, AlertCircle, Loader2, X } from 'lucide-react';
import Sidebar from '../components/layout/Sidebar';

export default function Upload() {
    const [file, setFile] = useState(null);
    const [dragActive, setDragActive] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');
    const { post } = useApi();
    const navigate = useNavigate();

    const handleDrag = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    }, []);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
            setError('');
        }
    }, []);

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setError('');
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setUploading(true);
        setError('');
        setResult(null);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await post('/api/upload', formData);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Upload failed');
            }

            // Check for ML error
            if (data.ml_error) {
                setError(`ML Service Error: ${data.ml_error}`);
                return;
            }

            // Store prediction data for dashboard
            if (data.prediction && data.prediction.detailed_predictions) {
                localStorage.setItem('csv_prediction', JSON.stringify({
                    data: data.prediction.detailed_predictions,
                    filename: file.name,
                    timestamp: Date.now()
                }));
            }

            setResult(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setUploading(false);
        }
    };

    const goToDashboard = () => {
        navigate('/?view=prediction');
    };

    return (
        <div className="flex min-h-screen bg-slate-900">
            <Sidebar />

            <main className="flex-1 p-8">
                <h1 className="text-2xl font-bold text-white mb-2">Upload Data</h1>
                <p className="text-slate-400 mb-8">Upload CSV files to generate demand predictions</p>

                {/* Upload Zone */}
                <div
                    className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all ${dragActive
                            ? 'border-emerald-500 bg-emerald-500/10'
                            : 'border-slate-700 hover:border-slate-600 bg-slate-800/30'
                        }`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                >
                    <input
                        type="file"
                        accept=".csv"
                        onChange={handleFileChange}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    />

                    <UploadIcon className="w-12 h-12 text-slate-500 mx-auto mb-4" />
                    <p className="text-lg text-white mb-2">
                        {file ? file.name : 'Drag & drop your CSV file here'}
                    </p>
                    <p className="text-sm text-slate-400">
                        or click to browse
                    </p>
                </div>

                {/* Selected File */}
                {file && (
                    <div className="mt-6 flex items-center justify-between bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                        <div className="flex items-center gap-3">
                            <FileText className="w-8 h-8 text-emerald-400" />
                            <div>
                                <p className="text-white font-medium">{file.name}</p>
                                <p className="text-sm text-slate-400">
                                    {(file.size / 1024).toFixed(1)} KB
                                </p>
                            </div>
                        </div>
                        <div className="flex items-center gap-2">
                            <button
                                onClick={() => setFile(null)}
                                className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
                            >
                                <X className="w-5 h-5 text-slate-400" />
                            </button>
                            <button
                                onClick={handleUpload}
                                disabled={uploading}
                                className="bg-emerald-500 hover:bg-emerald-600 disabled:bg-emerald-500/50 text-white px-6 py-2 rounded-lg font-medium flex items-center gap-2 transition-colors"
                            >
                                {uploading ? (
                                    <>
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Processing...
                                    </>
                                ) : (
                                    'Generate Predictions'
                                )}
                            </button>
                        </div>
                    </div>
                )}

                {/* Error */}
                {error && (
                    <div className="mt-6 p-4 bg-red-500/20 border border-red-500/50 rounded-xl flex items-center gap-3">
                        <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                        <p className="text-red-400">{error}</p>
                    </div>
                )}

                {/* Success Result */}
                {result && (
                    <div className="mt-6 p-6 bg-emerald-500/20 border border-emerald-500/50 rounded-xl">
                        <div className="flex items-center gap-3 mb-4">
                            <CheckCircle className="w-6 h-6 text-emerald-400" />
                            <h3 className="text-lg font-semibold text-white">Prediction Complete!</h3>
                        </div>

                        {result.prediction && (
                            <div className="grid grid-cols-2 gap-4 mb-4">
                                <div className="bg-slate-900/50 rounded-lg p-4">
                                    <p className="text-sm text-slate-400">Products Analyzed</p>
                                    <p className="text-2xl font-bold text-white">{result.prediction.count || 0}</p>
                                </div>
                                <div className="bg-slate-900/50 rounded-lg p-4">
                                    <p className="text-sm text-slate-400">File Status</p>
                                    <p className="text-2xl font-bold text-emerald-400">Success</p>
                                </div>
                            </div>
                        )}

                        <button
                            onClick={goToDashboard}
                            className="w-full bg-emerald-500 hover:bg-emerald-600 text-white py-3 rounded-lg font-medium transition-colors"
                        >
                            View Predictions on Dashboard
                        </button>
                    </div>
                )}

                {/* Instructions */}
                <div className="mt-8 bg-slate-800/30 border border-slate-700 rounded-xl p-6">
                    <h3 className="text-lg font-semibold text-white mb-4">CSV Format Requirements</h3>
                    <ul className="space-y-2 text-slate-400">
                        <li className="flex items-start gap-2">
                            <span className="text-emerald-400">•</span>
                            <span>Required column: <code className="text-emerald-400 bg-slate-900 px-1 rounded">sales_history</code></span>
                        </li>
                        <li className="flex items-start gap-2">
                            <span className="text-emerald-400">•</span>
                            <span>Optional column: <code className="text-emerald-400 bg-slate-900 px-1 rounded">description</code></span>
                        </li>
                        <li className="flex items-start gap-2">
                            <span className="text-emerald-400">•</span>
                            <span>Sales history should be a list of numbers (e.g., [10, 15, 20, ...])</span>
                        </li>
                    </ul>
                </div>
            </main>
        </div>
    );
}
