"use client";

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import api from '@/lib/api';
import axios from 'axios';
import { UploadCloud, CheckCircle2, XCircle, AlertCircle, Video, Play, Square, RefreshCw, Camera } from 'lucide-react';

interface ProductCategory {
  name: string;
  display_name?: string;
  classes?: string[];
  autoencoder_available?: boolean;
  classifier_available?: boolean;
}

interface InspectionResult {
  status: string;
  defect_type?: string;
  classification_confidence?: number;
  severity_level?: string;
  severity_score?: number;
  recommendation?: string;
  defect_region?: {
    x?: number;
    y?: number;
    width?: number;
    height?: number;
    orig_width?: number;
    orig_height?: number;
  };
}

interface VizData {
  heatmap_b64?: string;
}

interface ValidationIssue {
  msg?: string;
}

function apiDetailMessage(err: unknown, fallback: string): string {
  const detail = axios.isAxiosError(err)
    ? (err.response?.data as { detail?: unknown } | undefined)?.detail
    : undefined;
  if (Array.isArray(detail)) {
    return (
      detail
        .map((d) => (typeof d === 'object' && d !== null && 'msg' in d ? String((d as ValidationIssue).msg) : ''))
        .filter(Boolean)
        .join(', ') || fallback
    );
  }
  if (typeof detail === 'string' && detail) return detail;
  if (err instanceof Error && err.message) return err.message;
  return fallback;
}

const FALLBACK_CATEGORIES: string[] = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut'];

export default function InspectPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<'single' | 'stream'>('single');

  // Phase 2: product category selection (persisted, backend-driven).
  const [categories, setCategories] = useState<ProductCategory[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('visioninspect_category') || 'bottle';
    }
    return 'bottle';
  });

  // Single Upload State
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InspectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Phase 3: heatmap / visualization toggle.
  const [vizMode, setVizMode] = useState<'raw' | 'heatmap'>('raw');
  const [vizData, setVizData] = useState<VizData | null>(null);
  const [vizLoading, setVizLoading] = useState(false);
  const [vizOpacity, setVizOpacity] = useState(0.55);

  // Live Stream Simulation State
  const [streaming, setStreaming] = useState(false);
  const [streamFrameCount, setStreamFrameCount] = useState(0);
  const [lastStreamResult, setLastStreamResult] = useState<InspectionResult & { timestamp?: string } | null>(null);

  // Stream tab mode: real webcam (default) vs demo simulation
  const [streamMode, setStreamMode] = useState<'webcam' | 'simulation'>('webcam');

  // Real webcam state
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [webcamActive, setWebcamActive] = useState(false);
  const [webcamStarting, setWebcamStarting] = useState(false);
  const [webcamError, setWebcamError] = useState<string | null>(null);
  const [webcamResult, setWebcamResult] = useState<InspectionResult | null>(null);
  const [capturing, setCapturing] = useState(false);

  const stopWebcamTracks = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const stopWebcam = () => {
    stopWebcamTracks();
    setWebcamActive(false);
  };

  const startWebcam = async () => {
    setWebcamError(null);
    setWebcamStarting(true);
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Camera API not supported in this browser. Use HTTPS or localhost with a modern browser.');
      }
      // Stop any previous tracks before requesting a new stream.
      stopWebcamTracks();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {
          // play() may reject if autoplay is blocked; user gesture already occurred so ignore.
        });
      }
      setWebcamActive(true);
    } catch (err: unknown) {
      const e = err as { message?: string; name?: string };
      let msg = e?.message || 'Could not access the camera.';
      if (e?.name === 'NotAllowedError') {
        msg = 'Camera permission denied. Allow camera access in the browser address bar and try again.';
      } else if (e?.name === 'NotFoundError' || e?.name === 'OverconstrainedError') {
        msg = 'No camera device found. Connect a camera and try again.';
      } else if (e?.name === 'NotReadableError') {
        msg = 'Camera is already in use by another app. Close it and try again.';
      }
      setWebcamError(msg);
      stopWebcamTracks();
      setWebcamActive(false);
    } finally {
      setWebcamStarting(false);
    }
  };

  const captureFrame = async () => {
    const video = videoRef.current;
    if (!video || !webcamActive) {
      setWebcamError('Start the webcam before capturing a frame.');
      return;
    }
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      setWebcamError('Camera feed not ready yet. Wait a moment and try again.');
      return;
    }
    setCapturing(true);
    setWebcamError(null);
    try {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Canvas 2D context unavailable.');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob: Blob | null = await new Promise((resolve) =>
        canvas.toBlob(resolve, 'image/jpeg', 0.92)
      );
      if (!blob) throw new Error('Failed to capture frame from camera.');
      const frameFile = new File([blob], `webcam-${Date.now()}.jpg`, { type: 'image/jpeg' });
      const formData = new FormData();
      formData.append('file', frameFile);
      formData.append('category', selectedCategory);
      // Let axios set the multipart boundary automatically.
      const res = await api.post(`/inspections/upload?category=${encodeURIComponent(selectedCategory)}`, formData);
      setWebcamResult(res.data as InspectionResult);
    } catch (err: unknown) {
      setWebcamError(apiDetailMessage(err, 'Frame inspection failed'));
    } finally {
      setCapturing(false);
    }
  };

  // Release camera tracks on unmount.
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
    };
  }, []);

  // Release the camera when navigating away from the stream tab.
  useEffect(() => {
    if (activeTab !== 'stream') {
      stopWebcamTracks();
      queueMicrotask(() => setWebcamActive(false));
    }
  }, [activeTab]);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const openFilePicker = () => fileInputRef.current?.click();

  const handleFiles = (files: FileList | File[] | null) => {
    const selected = files?.[0];
    if (!selected) return;
    if (selected.size > 10 * 1024 * 1024) {
      setError(`"${selected.name}" exceeds 10MB size limit.`);
      return;
    }
    const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp'];
    const allowedExts = ['.jpg', '.jpeg', '.png', '.bmp'];
    const ext = '.' + (selected.name.split('.').pop() || '').toLowerCase();
    if (
      (selected.type && !allowedTypes.includes(selected.type)) ||
      (!selected.type && !allowedExts.includes(ext))
    ) {
      setError(`Unsupported format "${selected.name}". Use JPG, PNG or BMP.`);
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setFile(selected);
    const url = URL.createObjectURL(selected);
    setPreview(url);
    setResult(null);
    setError(null);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFiles(e.target.files);
    // Reset so selecting the same file again still fires onChange.
    e.target.value = '';
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    handleFiles(e.dataTransfer.files);
  };

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  useEffect(() => {
    const token = localStorage.getItem('token') || localStorage.getItem('access_token');
    if (!token) {
      router.push('/login');
      return;
    }
    // Phase 2: load categories for the selector (backend-driven).
    const loadCategories = async () => {
      try {
        const res = await api.get('/inspections/categories');
        const list = (res.data as { categories?: unknown } | undefined)?.categories;
        if (Array.isArray(list) && list.length > 0) {
          const typed = list as ProductCategory[];
          setCategories(typed);
          const names = typed.map((c) => c.name);
          const stored = localStorage.getItem('visioninspect_category');
          if (!stored || !names.includes(stored)) {
            const fallback = names.includes('bottle') ? 'bottle' : names[0];
            setSelectedCategory(fallback);
            localStorage.setItem('visioninspect_category', fallback);
          }
          return;
        }
      } catch {
        // Offline / unauthorized: fall back to static list below.
      }
      setCategories(
        FALLBACK_CATEGORIES.map((name: string) => ({ name, display_name: name, classes: [] }))
      );
    };
    loadCategories();
  }, [router]);

  const handleCategoryChange = (value: string) => {
    setSelectedCategory(value);
    try {
      localStorage.setItem('visioninspect_category', value);
    } catch {
      // storage unavailable (private mode) — selection still applies.
    }
    setResult(null);
    setWebcamResult(null);
    setVizData(null);
    setVizMode('raw');
  };

  const selectedCategoryInfo = categories.find((c) => c.name === selectedCategory);

  const handleInspect = async () => {
    if (!file) {
      // No file yet: open the picker instead of doing nothing.
      openFilePicker();
      return;
    }

    setLoading(true);
    setError(null);
    setVizData(null);
    setVizMode('raw');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('category', selectedCategory);

    try {
      // Let axios set multipart boundary automatically.
      const res = await api.post(`/inspections/upload?category=${encodeURIComponent(selectedCategory)}`, formData);
      setResult(res.data as InspectionResult);
    } catch (err: unknown) {
      setError(apiDetailMessage(err, 'Inspection failed'));
    } finally {
      setLoading(false);
    }
  };

  const handleVisualize = async () => {
    if (!file) {
      openFilePicker();
      return;
    }
    setVizLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('category', selectedCategory);
      const res = await api.post(`/inspections/visualize?category=${encodeURIComponent(selectedCategory)}`, formData);
      setVizData(res.data as VizData);
      setVizMode('heatmap');
    } catch (err: unknown) {
      setError(apiDetailMessage(err, 'Visualization failed'));
    } finally {
      setVizLoading(false);
    }
  };

  // Simulated Camera Stream effect
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;
    if (streaming) {
      interval = setInterval(() => {
        setStreamFrameCount(prev => prev + 1);
        // Simulate a real-time inspection output for stream demo
        const isPass = Math.random() > 0.3;
        const defects = ['broken_large', 'broken_small', 'contamination'];
        const defect = isPass ? 'good' : defects[Math.floor(Math.random() * defects.length)];
        
        setLastStreamResult({
          status: isPass ? 'PASS' : 'FAIL',
          defect_type: defect,
          classification_confidence: (0.85 + Math.random() * 0.14),
          severity_score: isPass ? Math.floor(Math.random() * 20) : Math.floor(50 + Math.random() * 45),
          timestamp: new Date().toLocaleTimeString()
        });
      }, 2000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [streaming]);

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header and Tab Switcher */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-3xl font-bold text-white tracking-wide">Surface Inspection Suite</h2>
          <p className="text-slate-400 text-sm mt-1">Real-time PyTorch ML anomaly & defect classification.</p>
        </div>
        
        <div className="flex bg-slate-900/80 p-1.5 rounded-xl border border-slate-700/60">
          <button
            onClick={() => setActiveTab('single')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition flex items-center gap-2 ${
              activeTab === 'single'
                ? 'bg-blue-600 text-white shadow-[0_0_12px_rgba(37,99,235,0.4)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <UploadCloud size={18} /> Single Image
          </button>
          <button
            onClick={() => setActiveTab('stream')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition flex items-center gap-2 ${
              activeTab === 'stream'
                ? 'bg-blue-600 text-white shadow-[0_0_12px_rgba(37,99,235,0.4)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Video size={18} /> Camera Stream
          </button>
        </div>
      </div>

      {activeTab === 'single' ? (
        <div className="glass-card p-8 rounded-2xl">
          {/* Phase 2: product category selector */}
          <div className="flex flex-col sm:flex-row sm:items-center gap-3 mb-6 p-4 rounded-xl bg-slate-900/60 border border-slate-700/60">
            <label htmlFor="product-category" className="text-sm font-semibold text-slate-200">
              Product category
            </label>
            <select
              id="product-category"
              value={selectedCategory}
              onChange={(e) => handleCategoryChange(e.target.value)}
              className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm min-w-48"
            >
              {(categories.length > 0 ? categories : FALLBACK_CATEGORIES.map((name: string) => ({ name, display_name: name }))).map((c: ProductCategory) => (
                <option key={c.name} value={c.name}>
                  {c.display_name || c.name}
                  {c.autoencoder_available === false || c.classifier_available === false ? ' (no model)' : ''}
                </option>
              ))}
            </select>
            {selectedCategoryInfo?.classes && selectedCategoryInfo.classes.length > 0 && (
              <span className="text-xs text-slate-400">
                Defects: {selectedCategoryInfo.classes.join(', ')}
              </span>
            )}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Uploader Section */}
            <div className="space-y-4">
              <div
                role="button"
                tabIndex={0}
                aria-label="Upload product image"
                onClick={openFilePicker}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    openFilePicker();
                  }
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
                className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer group ${
                  dragActive
                    ? 'border-blue-400 bg-blue-500/10'
                    : 'border-slate-600 hover:bg-slate-800/30'
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".jpg,.jpeg,.png,.bmp"
                  className="hidden"
                  onChange={handleFileChange}
                />
                {preview ? (
                  <div className="space-y-3">
                    <div className="relative aspect-video rounded-lg overflow-hidden flex items-center justify-center bg-slate-900">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={vizMode === 'heatmap' && vizData?.heatmap_b64 ? `data:image/png;base64,${vizData.heatmap_b64}` : preview} alt="Preview" className="max-h-full max-w-full object-contain" />
                      {vizMode === 'heatmap' && vizData?.heatmap_b64 && preview && (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={`data:image/png;base64,${vizData.heatmap_b64}`}
                          alt="Anomaly heatmap"
                          className="absolute inset-0 h-full w-full object-contain pointer-events-none"
                          style={{ opacity: vizOpacity }}
                        />
                      )}
                      {/* Phase 1: defect bounding-box overlay (backend defect_region, original-size coords). */}
                      {vizMode === 'raw' && result?.defect_region && (() => {
                        const r = result.defect_region;
                        const ow = r.orig_width || 224;
                        const oh = r.orig_height || 224;
                        if (r.x == null || r.y == null) return null;
                        return (
                          <svg
                            viewBox={`0 0 ${ow} ${oh}`}
                            className="absolute inset-0 h-full w-full pointer-events-none"
                            preserveAspectRatio="xMidYMid meet"
                          >
                            <rect
                              x={r.x}
                              y={r.y}
                              width={r.width}
                              height={r.height}
                              fill="none"
                              stroke={result.status === 'PASS' ? '#34d399' : '#ef4444'}
                              strokeWidth={Math.max(2, ow / 200)}
                            />
                            <text
                              x={r.x}
                              y={Math.max(12, r.y - 4)}
                              fill={result.status === 'PASS' ? '#34d399' : '#ef4444'}
                              fontSize={Math.max(10, ow / 40)}
                              fontWeight="bold"
                            >
                              {String(result.defect_type || '').replaceAll('_', ' ')}
                              {typeof result.classification_confidence === 'number'
                                ? ` ${(result.classification_confidence * 100).toFixed(0)}%`
                                : ''}
                            </text>
                          </svg>
                        );
                      })()}
                    </div>
                    {/* Phase 3: Raw ↔ Heatmap toggle + opacity */}
                    {(result || vizData) && (
                      <div className="flex items-center gap-3 text-xs text-slate-300">
                        <div className="flex bg-slate-800 p-1 rounded-lg border border-slate-700">
                          <button
                            type="button"
                            onClick={() => setVizMode('raw')}
                            className={`px-3 py-1 rounded-md font-semibold ${vizMode === 'raw' ? 'bg-blue-600 text-white' : 'text-slate-400'}`}
                          >
                            Raw + Box
                          </button>
                          <button
                            type="button"
                            onClick={handleVisualize}
                            disabled={vizLoading}
                            className={`px-3 py-1 rounded-md font-semibold ${vizMode === 'heatmap' ? 'bg-blue-600 text-white' : 'text-slate-400'}`}
                          >
                            {vizLoading ? 'Loading…' : 'Heatmap'}
                          </button>
                        </div>
                        {vizMode === 'heatmap' && (
                          <label className="flex items-center gap-2">
                            Opacity
                            <input
                              type="range"
                              min="0"
                              max="1"
                              step="0.05"
                              value={vizOpacity}
                              onChange={(e) => setVizOpacity(parseFloat(e.target.value))}
                              className="accent-blue-500"
                            />
                          </label>
                        )}
                      </div>
                    )}
                    {file && (
                      <div className="flex items-center justify-between gap-2 text-sm">
                        <span className="text-slate-300 truncate" title={file.name}>
                          {file.name} ({(file.size / 1024).toFixed(0)} KB)
                        </span>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            openFilePicker();
                          }}
                          className="shrink-0 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
                        >
                          Change
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="py-12">
                    <div className="w-16 h-16 mx-auto rounded-full bg-blue-500/10 flex items-center justify-center text-blue-400 group-hover:scale-110 transition-transform">
                      <UploadCloud size={32} />
                    </div>
                    <h3 className="mt-4 text-lg font-medium text-slate-200">
                      {dragActive ? 'Drop image to inspect' : 'Drag & drop product image'}
                    </h3>
                    <p className="mt-1 text-sm text-slate-400">JPG, PNG, or BMP (max 10MB)</p>
                    <p className="mt-3 text-sm font-semibold text-blue-400">or click to browse files</p>
                  </div>
                )}
              </div>

              <button
                onClick={handleInspect}
                disabled={loading}
                title={!file ? 'Select an image first' : 'Run inspection'}
                className={`w-full py-4 rounded-xl font-bold text-lg transition-all ${
                  loading ? 'bg-blue-600/50 text-white cursor-wait' :
                  'bg-blue-600 hover:bg-blue-500 text-white shadow-[0_0_20px_rgba(37,99,235,0.4)]'
                }`}
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Analyzing with AI...
                  </span>
                ) : (file ? 'Run AI Inspection' : 'Select Image to Inspect')}
              </button>
              
              {error && (
                <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 flex items-start gap-3">
                  <AlertCircle className="shrink-0" />
                  <p>{error}</p>
                </div>
              )}
            </div>

            {/* Results Section */}
            <div className="glass-panel p-6 rounded-xl border border-slate-700/50 relative overflow-hidden">
              {result && (
                 <div className={`absolute -top-20 -right-20 w-64 h-64 rounded-full blur-[100px] opacity-20 ${
                   result.status === 'PASS' ? 'bg-emerald-500' : 'bg-red-500'
                 }`}></div>
              )}

              <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                Inspection Results
              </h3>
              
              {!result ? (
                <div className="h-48 flex flex-col items-center justify-center text-slate-500 border border-slate-800 rounded-xl border-dashed">
                  <p>Upload an image and run inspection to see AI results here.</p>
                </div>
              ) : (
                <div className="space-y-6 relative z-10">
                  <div className="flex items-center justify-between p-4 rounded-xl bg-slate-800/50 border border-slate-700">
                    <span className="text-slate-300">Decision</span>
                    <div className={`flex items-center gap-2 font-bold text-lg px-4 py-1 rounded-full ${
                      result.status === 'PASS' ? 'text-emerald-400 bg-emerald-500/10' : 'text-red-400 bg-red-500/10'
                    }`}>
                      {result.status === 'PASS' ? <CheckCircle2 size={20} /> : <XCircle size={20} />}
                      {result.status}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 rounded-xl bg-slate-800/50">
                      <p className="text-sm text-slate-400 mb-1">Defect Type</p>
                      <p className="font-bold text-white capitalize">{(result.defect_type ?? 'unknown').replaceAll('_', ' ')}</p>
                    </div>
                    <div className="p-4 rounded-xl bg-slate-800/50">
                      <p className="text-sm text-slate-400 mb-1">AI Confidence</p>
                      <p className="font-bold text-white">{typeof result.classification_confidence === 'number' ? (result.classification_confidence * 100).toFixed(1) + '%' : '-'}</p>
                    </div>
                    <div className="p-4 rounded-xl bg-slate-800/50">
                      <p className="text-sm text-slate-400 mb-1">Severity Level</p>
                      <p className={`font-bold ${
                        result.severity_level === 'Critical' ? 'text-red-500' :
                        result.severity_level === 'High' ? 'text-orange-400' :
                        result.severity_level === 'Medium' ? 'text-amber-400' : 'text-emerald-400'
                      }`}>{result.severity_level ?? '-'}</p>
                    </div>
                    <div className="p-4 rounded-xl bg-slate-800/50">
                      <p className="text-sm text-slate-400 mb-1">Severity Score</p>
                      <p className="font-bold text-white">{result.severity_score ?? 0}/100</p>
                    </div>
                  </div>

                  {result.recommendation && (
                    <div className="p-4 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-100">
                      <p className="text-sm font-bold text-blue-400 mb-1">Recommendation</p>
                      <p className="text-sm leading-relaxed">{result.recommendation}</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      ) : (
        /* Camera Stream Feed: real webcam (default) + demo simulation */
        <div className="glass-card p-8 rounded-2xl space-y-6">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${(streamMode === 'webcam' && webcamActive) || (streamMode === 'simulation' && streaming) ? 'bg-emerald-500 animate-pulse' : 'bg-slate-600'}`}></div>
              <h3 className="text-xl font-bold text-white">
                {streamMode === 'webcam' ? 'Live Webcam Inspection' : 'Simulation (Demo Only)'}
              </h3>
            </div>

            <div className="flex bg-slate-900/80 p-1 rounded-xl border border-slate-700/60">
              <button
                onClick={() => { setStreaming(false); setStreamMode('webcam'); }}
                className={`px-3 py-1.5 rounded-lg text-sm font-semibold transition flex items-center gap-2 ${
                  streamMode === 'webcam' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Camera size={16} /> Webcam
              </button>
              <button
                onClick={() => { stopWebcam(); setStreamMode('simulation'); }}
                className={`px-3 py-1.5 rounded-lg text-sm font-semibold transition flex items-center gap-2 ${
                  streamMode === 'simulation' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Play size={16} /> Simulation
              </button>
            </div>
          </div>

          {streamMode === 'webcam' ? (
            <div className="space-y-6">
              <p className="text-sm text-slate-400">
                Real camera feed via <span className="font-mono">getUserMedia</span>. Captured frames are sent to
                the real backend at <span className="font-mono">POST /inspections/upload</span> for ML inference.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="md:col-span-2 glass-panel p-4 rounded-xl bg-slate-950/80 border border-slate-800 relative aspect-video flex items-center justify-center overflow-hidden">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className={`w-full h-full object-cover rounded-lg ${webcamActive ? '' : 'hidden'}`}
                  />
                  {!webcamActive && (
                    <div className="text-center text-slate-500">
                      <Video size={48} className="mx-auto mb-2 opacity-50" />
                      <p>Webcam offline. Click &quot;Start Webcam&quot; to enable your camera.</p>
                    </div>
                  )}
                </div>

                <div className="glass-panel p-6 rounded-xl border border-slate-800 space-y-4">
                  <h4 className="text-lg font-bold text-white border-b border-slate-800 pb-2">Webcam Controls</h4>
                  {!webcamActive ? (
                    <button
                      onClick={startWebcam}
                      disabled={webcamStarting}
                      className="w-full px-5 py-2.5 rounded-xl font-bold text-sm bg-emerald-600 hover:bg-emerald-500 text-white disabled:opacity-50 flex items-center justify-center gap-2"
                    >
                      <Play size={16} /> {webcamStarting ? 'Starting…' : 'Start Webcam'}
                    </button>
                  ) : (
                    <div className="space-y-3">
                      <button
                        onClick={captureFrame}
                        disabled={capturing}
                        className="w-full px-5 py-2.5 rounded-xl font-bold text-sm bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-50 flex items-center justify-center gap-2"
                      >
                        <Camera size={16} /> {capturing ? 'Inspecting…' : 'Capture Frame & Inspect'}
                      </button>
                      <button
                        onClick={stopWebcam}
                        className="w-full px-5 py-2.5 rounded-xl font-bold text-sm bg-red-600 hover:bg-red-500 text-white flex items-center justify-center gap-2"
                      >
                        <Square size={16} /> Stop Webcam
                      </button>
                    </div>
                  )}

                  {webcamError && (
                    <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm flex items-start gap-2">
                      <AlertCircle className="shrink-0 mt-0.5" size={16} />
                      <p>{webcamError}</p>
                    </div>
                  )}

                  {!webcamResult ? (
                    <p className="text-slate-500 text-sm text-center">Capture a frame to see real ML results here.</p>
                  ) : (
                    <div className="space-y-3">
                      <div className={`flex items-center gap-2 font-bold text-lg px-4 py-1 rounded-full w-max ${
                        webcamResult.status === 'PASS' ? 'text-emerald-400 bg-emerald-500/10' : 'text-red-400 bg-red-500/10'
                      }`}>
                        {webcamResult.status === 'PASS' ? <CheckCircle2 size={20} /> : <XCircle size={20} />}
                        {webcamResult.status}
                      </div>
                      <div>
                        <span className="text-xs text-slate-400 uppercase">Defect Type</span>
                        <p className="text-white font-semibold capitalize">{String(webcamResult.defect_type ?? 'unknown').replaceAll('_', ' ')}</p>
                      </div>
                      <div>
                        <span className="text-xs text-slate-400 uppercase">AI Confidence</span>
                        <p className="text-white font-semibold">
                          {typeof webcamResult.classification_confidence === 'number'
                            ? `${(webcamResult.classification_confidence * 100).toFixed(1)}%`
                            : '-'}
                        </p>
                      </div>
                      <div>
                        <span className="text-xs text-slate-400 uppercase">Severity</span>
                        <p className="text-white font-semibold">
                          {webcamResult.severity_level ?? '-'} ({webcamResult.severity_score ?? 0}/100)
                        </p>
                      </div>
                      {webcamResult.recommendation && (
                        <p className="text-sm text-blue-100 bg-blue-500/10 border border-blue-500/20 rounded-xl p-3">{webcamResult.recommendation}</p>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="flex items-center justify-between flex-wrap gap-3">
                <div className="flex items-center gap-3">
                  <div className={`w-3 h-3 rounded-full ${streaming ? 'bg-emerald-500 animate-pulse' : 'bg-slate-600'}`}></div>
                  <h4 className="text-lg font-bold text-white">Simulation (Demo Only)</h4>
                </div>

                <button
                  onClick={() => setStreaming(!streaming)}
                  className={`px-5 py-2.5 rounded-xl font-bold text-sm transition flex items-center gap-2 ${
                    streaming
                      ? 'bg-red-600 hover:bg-red-500 text-white shadow-[0_0_15px_rgba(239,68,68,0.4)]'
                      : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-[0_0_15px_rgba(16,185,129,0.4)]'
                  }`}
                >
                  {streaming ? <><Square size={16} /> Stop Feed</> : <><Play size={16} /> Start Simulated Stream</>}
                </button>
              </div>

              <p className="text-sm text-amber-300/90 bg-amber-500/10 border border-amber-500/20 rounded-xl p-3">
                Disclaimer: this simulation generates random PASS/FAIL values locally for UI demo purposes only.
                It does not run ML inference and is not connected to a camera or the backend.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Viewfinder */}
                <div className="md:col-span-2 glass-panel p-4 rounded-xl bg-slate-950/80 border border-slate-800 relative aspect-video flex items-center justify-center overflow-hidden">
                  {streaming ? (
                    <div className="w-full h-full relative flex items-center justify-center">
                      <div className="absolute inset-0 bg-gradient-to-t from-slate-950 via-transparent to-transparent z-10"></div>

                      {/* Simulated laser scan line animation */}
                      <div className="absolute inset-x-0 h-0.5 bg-blue-500 shadow-[0_0_15px_#3b82f6] animate-pulse top-1/2"></div>

                      <div className="text-center z-20 space-y-2">
                        <RefreshCw className="animate-spin text-blue-400 mx-auto" size={36} />
                        <p className="text-slate-300 font-semibold text-sm">Simulating Frames (2.0 FPS)</p>
                        <p className="text-xs text-slate-500 font-mono">Frames Processed: {streamFrameCount}</p>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center text-slate-500">
                      <Video size={48} className="mx-auto mb-2 opacity-50" />
                      <p>Simulation offline. Click &quot;Start Simulated Stream&quot; to launch the demo feed.</p>
                    </div>
                  )}
                </div>

                {/* Stream Telemetry */}
                <div className="glass-panel p-6 rounded-xl border border-slate-800 space-y-4">
                  <h4 className="text-lg font-bold text-white border-b border-slate-800 pb-2">Simulated Telemetry</h4>

                  {!lastStreamResult ? (
                    <p className="text-slate-500 text-sm py-8 text-center">No telemetry frame received yet.</p>
                  ) : (
                    <div className="space-y-4">
                      <div>
                        <span className="text-xs text-slate-400 uppercase">Last Frame Status</span>
                        <div className={`mt-1 font-bold text-xl px-3 py-1 rounded-lg w-max ${
                          lastStreamResult.status === 'PASS' ? 'text-emerald-400 bg-emerald-500/10' : 'text-red-400 bg-red-500/10'
                        }`}>
                          {lastStreamResult.status}
                        </div>
                      </div>

                      <div>
                        <span className="text-xs text-slate-400 uppercase">Defect Classification</span>
                        <p className="text-white font-semibold capitalize mt-0.5">{(lastStreamResult.defect_type ?? 'unknown').replace('_', ' ')}</p>
                      </div>

                      <div>
                        <span className="text-xs text-slate-400 uppercase">Severity Score</span>
                        <div className="w-full bg-slate-800 rounded-full h-2 mt-1.5 overflow-hidden">
                          <div
                            className={`h-full ${(lastStreamResult.severity_score ?? 0) > 50 ? 'bg-red-500' : 'bg-emerald-500'}`}
                            style={{ width: `${lastStreamResult.severity_score ?? 0}%` }}
                          ></div>
                        </div>
                      </div>

                      <div>
                        <span className="text-xs text-slate-400 uppercase">Timestamp</span>
                        <p className="text-slate-300 font-mono text-xs mt-0.5">{lastStreamResult.timestamp}</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
