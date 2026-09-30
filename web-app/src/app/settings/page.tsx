"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import api from '@/lib/api';
import axios from 'axios';
import { Sliders, Save, RefreshCw, ShieldAlert, CheckCircle2, Info } from 'lucide-react';

interface ValidationIssue {
  msg?: string;
}

function settingsErrorMessage(err: unknown, fallback: string): string {
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

interface SettingsPayload {
  blur_threshold?: unknown;
  anomaly_threshold?: unknown;
  classification_confidence_threshold?: unknown;
  critical_severity_cutoff?: unknown;
  high_severity_cutoff?: unknown;
  medium_severity_cutoff?: unknown;
}

export default function SettingsPage() {
  const router = useRouter();
  // Phase 1 unified defaults: blur 20, anomaly 90, critical 80/high 60/medium 40.
  const [blurThreshold, setBlurThreshold] = useState<number>(20);
  const [anomalyThreshold, setAnomalyThreshold] = useState<number>(90);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.5);
  const [criticalCutoff, setCriticalCutoff] = useState<number>(80);
  const [highCutoff, setHighCutoff] = useState<number>(60);
  const [mediumCutoff, setMediumCutoff] = useState<number>(40);

  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  const fetchSettings = async () => {
    setLoading(true);
    try {
      const res = await api.get('/inspections/config/settings');
      const s = (res.data as { settings?: SettingsPayload } | undefined)?.settings;
      if (s) {
        if (s.blur_threshold != null) setBlurThreshold(Number(s.blur_threshold));
        if (s.anomaly_threshold != null) setAnomalyThreshold(Number(s.anomaly_threshold));
        if (s.classification_confidence_threshold != null) setConfidenceThreshold(Number(s.classification_confidence_threshold));
        if (s.critical_severity_cutoff != null) setCriticalCutoff(Number(s.critical_severity_cutoff));
        if (s.high_severity_cutoff != null) setHighCutoff(Number(s.high_severity_cutoff));
        if (s.medium_severity_cutoff != null) setMediumCutoff(Number(s.medium_severity_cutoff));
      }
    } catch (err) {
      console.error('Failed to load settings', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const role = localStorage.getItem('role');
    const token = localStorage.getItem('token') || localStorage.getItem('access_token');
    if (!token) {
      router.push('/login');
      return;
    }
    if (role !== 'supervisor' && role !== 'admin') {
      router.push('/');
      return;
    }
    // Defer fetch so state updates happen outside the synchronous effect body.
    queueMicrotask(() => {
      fetchSettings();
    });
  }, [router]);

  useEffect(() => {
    if (!toastMessage) return;
    const t = setTimeout(() => setToastMessage(null), 3500);
    return () => clearTimeout(t);
  }, [toastMessage]);

  const handleSave = async () => {
    setSaving(true);
    setFormError(null);
    if (!(criticalCutoff > highCutoff && highCutoff > mediumCutoff)) {
      setFormError('Cutoffs must satisfy: Critical > High > Medium.');
      setSaving(false);
      return;
    }
    if ([blurThreshold, anomalyThreshold, criticalCutoff, highCutoff, mediumCutoff].some((v) => Number.isNaN(v) || v < 0 || v > 300)) {
      setFormError('Thresholds must be numbers between 0 and 300.');
      setSaving(false);
      return;
    }
    if (Number.isNaN(confidenceThreshold) || confidenceThreshold < 0 || confidenceThreshold > 1) {
      setFormError('Confidence threshold must be between 0 and 1.');
      setSaving(false);
      return;
    }
    try {
      await api.post('/inspections/config/settings', {
        blur_threshold: blurThreshold,
        anomaly_threshold: anomalyThreshold,
        classification_confidence_threshold: confidenceThreshold,
        critical_severity_cutoff: criticalCutoff,
        high_severity_cutoff: highCutoff,
        medium_severity_cutoff: mediumCutoff
      });
      setToastMessage('Threshold settings saved successfully!');
    } catch (err: unknown) {
      setFormError(settingsErrorMessage(err, 'Failed to save settings'));
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-wide flex items-center gap-3">
            <Sliders className="text-blue-400" size={32} />
            Quality Threshold Settings
          </h1>
          <p className="text-slate-400 mt-1">Configure AI decision boundaries, blur sensitivity, and severity cutoffs.</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={fetchSettings}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl transition flex items-center gap-2 text-sm border border-slate-700"
          >
            <RefreshCw size={16} /> Reset
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-medium rounded-xl transition shadow-[0_0_15px_rgba(37,99,235,0.4)] flex items-center gap-2"
          >
            <Save size={18} /> {saving ? 'Saving...' : 'Save Settings'}
          </button>
        </div>
      </div>

      {toastMessage && (
        <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 flex items-center gap-3">
          <CheckCircle2 size={20} />
          <span>{toastMessage}</span>
        </div>
      )}

      {formError && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 flex items-center gap-3">
          <ShieldAlert size={20} />
          <span>{formError}</span>
        </div>
      )}

      {/* Image Quality Controls */}
      <div className="glass-panel p-6 space-y-6">
        <div className="border-b border-slate-700/50 pb-4">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Info className="text-blue-400" size={20} />
            Pre-Inference Image Quality Gate
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">Images failing quality checks are auto-rejected before reaching AI inference.</p>
        </div>

        <div className="space-y-4">
          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium text-slate-200">
                Blur Detection Threshold (Laplacian Variance)
              </label>
              <span className="text-sm font-mono text-blue-400 bg-blue-500/10 px-3 py-1 rounded-lg border border-blue-500/20">
                {blurThreshold}
              </span>
            </div>
            <input
              type="range"
              min="20"
              max="300"
              step="5"
              value={blurThreshold}
              onChange={(e) => setBlurThreshold(parseFloat(e.target.value))}
              className="w-full accent-blue-500 h-2 bg-slate-800 rounded-lg cursor-pointer"
            />
            <p className="text-xs text-slate-400 mt-1">Lower values allow blurrier images; higher values require sharper focus.</p>
          </div>
        </div>
      </div>

      {/* AI Anomaly Thresholds */}
      <div className="glass-panel p-6 space-y-6">
        <div className="border-b border-slate-700/50 pb-4">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <ShieldAlert className="text-amber-400" size={20} />
            AI Anomaly & Defect Sensitivity
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">Adjust how aggressively the AI flags non-conforming surface defects.</p>
        </div>

        <div className="space-y-6">
          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium text-slate-200">
                Anomaly Detection Sensitivity Threshold
              </label>
              <span className="text-sm font-mono text-amber-400 bg-amber-500/10 px-3 py-1 rounded-lg border border-amber-500/20">
                {anomalyThreshold}
              </span>
            </div>
            <input
              type="range"
              min="50"
              max="150"
              step="1"
              value={anomalyThreshold}
              onChange={(e) => setAnomalyThreshold(parseFloat(e.target.value))}
              className="w-full accent-amber-500 h-2 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium text-slate-200">
                Classification Confidence Threshold
              </label>
              <span className="text-sm font-mono text-emerald-400 bg-emerald-500/10 px-3 py-1 rounded-lg border border-emerald-500/20">
                {confidenceThreshold}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
              className="w-full accent-emerald-500 h-2 bg-slate-800 rounded-lg cursor-pointer"
            />
            <p className="text-xs text-slate-400 mt-1">Minimum classifier confidence to accept a defect call (DB-wired, Phase 1).</p>
          </div>

          {/* Severity Cutoffs */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <span className="text-xs font-semibold text-rose-400 uppercase tracking-wider">Critical Severity Cutoff</span>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  value={criticalCutoff}
                  onChange={(e) => setCriticalCutoff(Number(e.target.value))}
                  className="w-full px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm"
                />
                <span className="text-slate-400 text-xs">%</span>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <span className="text-xs font-semibold text-amber-400 uppercase tracking-wider">High Severity Cutoff</span>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  value={highCutoff}
                  onChange={(e) => setHighCutoff(Number(e.target.value))}
                  className="w-full px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm"
                />
                <span className="text-slate-400 text-xs">%</span>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <span className="text-xs font-semibold text-blue-400 uppercase tracking-wider">Medium Severity Cutoff</span>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  value={mediumCutoff}
                  onChange={(e) => setMediumCutoff(Number(e.target.value))}
                  className="w-full px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm"
                />
                <span className="text-slate-400 text-xs">%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
