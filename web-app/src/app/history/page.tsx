"use client";

import { useCallback, useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import api from '@/lib/api';
import axios from 'axios';
import { format } from 'date-fns';
import { AlertCircle, CheckCircle2, Download, ShieldCheck, Filter, Search, Edit3, X } from 'lucide-react';

interface Inspection {
  id: number;
  created_at?: string;
  status: string;
  defect_type?: string;
  severity_level?: string;
  severity_score?: number;
  confidence?: number;
  is_overridden?: boolean;
  override_reason?: string;
  product_category?: string;
}

interface ValidationIssue {
  msg?: string;
}

function detailMessage(err: unknown, fallback: string): string {
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

export default function HistoryPage() {
  const router = useRouter();
  const [history, setHistory] = useState<Inspection[]>([]);
  const [loading, setLoading] = useState(true);
  const [userRole, setUserRole] = useState<string | null>(() =>
    typeof window === 'undefined' ? null : localStorage.getItem('role')
  );

  // Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [categoryFilter, setCategoryFilter] = useState('ALL');

  // Override Modal state
  const [selectedInspection, setSelectedInspection] = useState<Inspection | null>(null);
  const [overrideStatus, setOverrideStatus] = useState<string>('PASS');
  const [overrideReason, setOverrideReason] = useState<string>('');
  const [submittingOverride, setSubmittingOverride] = useState(false);

  const [fetchError, setFetchError] = useState<string | null>(null);

  const fetchHistory = useCallback(async (category: string) => {
    setLoading(true);
    setFetchError(null);
    try {
      const params = new URLSearchParams({ limit: '500' });
      const cat = category !== 'ALL' ? category : '';
      if (cat) params.set('category', cat);
      const res = await api.get(`/inspections/history?${params.toString()}`);
      setHistory(Array.isArray(res.data) ? (res.data as Inspection[]) : []);
    } catch (err: unknown) {
      console.error("Failed to fetch history", err);
      setFetchError(detailMessage(err, 'Failed to load history'));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const token = localStorage.getItem('token') || localStorage.getItem('access_token');
    if (!token) {
      router.push('/login');
      return;
    }
    // Defer fetch so state updates happen outside the synchronous effect body.
    queueMicrotask(() => {
      fetchHistory(categoryFilter);
    });
    const syncRole = () => {
      setUserRole(
        typeof window === 'undefined' ? null : localStorage.getItem('role')
      );
    };
    window.addEventListener('storage', syncRole);
    window.addEventListener('focus', syncRole);
    return () => {
      window.removeEventListener('storage', syncRole);
      window.removeEventListener('focus', syncRole);
    };
  }, [router, fetchHistory, categoryFilter]);

  const handleExportCSV = () => {
    if (history.length === 0) return;
    const headers = ["ID", "Timestamp", "Status", "Defect Type", "Severity Level", "Severity Score", "Confidence", "Overridden", "Override Reason"];
    const esc = (v: unknown) => `"${String(v ?? '').replace(/"/g, '""')}"`;
    const safeDate = (d: unknown) => {
      try {
        return d ? format(new Date(d as string), 'yyyy-MM-dd HH:mm:ss') : '';
      } catch {
        return String(d ?? '');
      }
    };
    const rows = history.map(item => [
      item.id,
      safeDate(item.created_at),
      item.status,
      item.defect_type || '',
      item.severity_level || '',
      item.severity_score ?? 0,
      typeof item.confidence === 'number' ? (item.confidence * 100).toFixed(1) + '%' : '',
      item.is_overridden ? 'Yes' : 'No',
      esc(item.override_reason || '')
    ]);

    const csvContent = [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `VisionInspect_History_${format(new Date(), 'yyyyMMdd_HHmm')}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleOpenOverride = (item: Inspection) => {
    setSelectedInspection(item);
    setOverrideStatus(item.status === 'PASS' ? 'FAIL' : 'PASS');
    setOverrideReason('');
  };

  const handleSubmitOverride = async () => {
    if (!selectedInspection || overrideReason.trim().length < 5) {
      setFetchError('Please enter a valid override reason (min 5 chars).');
      return;
    }

    setSubmittingOverride(true);
    try {
      await api.put(`/inspections/${selectedInspection.id}/override`, {
        new_status: overrideStatus,
        reason: overrideReason
      });
      setSelectedInspection(null);
      fetchHistory(categoryFilter);
    } catch (err: unknown) {
      setFetchError(detailMessage(err, 'Failed to submit override'));
    } finally {
      setSubmittingOverride(false);
    }
  };

  const filteredHistory = history.filter(item => {
    const matchesSearch = String(item.id ?? '').includes(searchQuery) ||
      (item.defect_type || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
      (item.severity_level || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
      (item.product_category || '').toLowerCase().includes(searchQuery.toLowerCase());

    const matchesStatus = statusFilter === 'ALL' || item.status === statusFilter;
    const matchesCategory = categoryFilter === 'ALL' ||
      (item.product_category || 'bottle') === categoryFilter;

    return matchesSearch && matchesStatus && matchesCategory;
  });

  const distinctCategories = Array.from(
    new Set(history.map((h) => h.product_category || 'bottle'))
  );

  const safeFormat = (d: unknown, fmt: string) => {
    try {
      return d ? format(new Date(d as string), fmt) : '-';
    } catch {
      return '-';
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="glass-card p-6 md:p-8 rounded-2xl space-y-6">
        {/* Header and Controls */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
          <div>
            <h2 className="text-2xl font-bold text-white flex items-center gap-3">
              Inspection Log History
            </h2>
            <p className="text-slate-400 text-sm mt-1">Audit log of automated surface defect inspections.</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={handleExportCSV}
              disabled={history.length === 0}
              className="px-4 py-2 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 border border-blue-500/30 rounded-xl transition flex items-center gap-2 text-sm font-medium"
            >
              <Download size={16} /> Export CSV
            </button>
          </div>
        </div>

        {fetchError && (
          <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
            {fetchError}
          </div>
        )}

        {/* Filter Bar */}
        <div className="flex flex-col sm:flex-row gap-4 justify-between items-center">
          <div className="relative w-full sm:w-72">
            <Search className="absolute left-3 top-2.5 text-slate-400" size={18} />
            <input
              type="text"
              placeholder="Search by ID, type..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-slate-900/60 border border-slate-700/60 rounded-xl text-white text-sm focus:outline-none focus:border-blue-500"
            />
          </div>

          <div className="flex items-center gap-2 w-full sm:w-auto">
            <Filter size={16} className="text-slate-400" />
            <span className="text-sm text-slate-400">Filter:</span>
            <div className="flex bg-slate-900/60 p-1 rounded-xl border border-slate-800">
              {['ALL', 'PASS', 'FAIL'].map((status) => (
                <button
                  key={status}
                  onClick={() => setStatusFilter(status)}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold transition ${
                    statusFilter === status
                      ? 'bg-blue-600 text-white shadow'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  {status}
                </button>
              ))}
            </div>
            <select
              aria-label="Filter by product category"
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="px-3 py-1.5 bg-slate-900/60 border border-slate-800 rounded-xl text-xs text-slate-200"
            >
              <option value="ALL">All products</option>
              {distinctCategories.map((c: string) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Table */}
        {loading ? (
          <div className="py-20 flex justify-center">
            <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
          </div>
        ) : filteredHistory.length === 0 ? (
          <div className="py-20 text-center text-slate-500">
            No matching inspection records found.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-700/50 text-slate-400 text-sm">
                  <th className="py-4 px-4 font-medium">ID</th>
                  <th className="py-4 px-4 font-medium">Date & Time</th>
                  <th className="py-4 px-4 font-medium">Decision</th>
                  <th className="py-4 px-4 font-medium">Defect Type</th>
                  <th className="py-4 px-4 font-medium">Severity</th>
                  <th className="py-4 px-4 font-medium text-right">Confidence</th>
                  {(userRole === 'supervisor' || userRole === 'admin') && (
                    <th className="py-4 px-4 font-medium text-center">Action</th>
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {filteredHistory.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-4 px-4 text-slate-300 font-mono">#{item.id}</td>
                    <td className="py-4 px-4 text-slate-300 text-sm">
                      {safeFormat(item.created_at, 'MMM dd, yyyy HH:mm')}
                    </td>
                    <td className="py-4 px-4">
                      <div className="flex flex-col gap-1">
                        <div className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold w-max ${
                          item.status === 'PASS' ? 'text-emerald-400 bg-emerald-500/10' : 'text-red-400 bg-red-500/10'
                        }`}>
                          {item.status === 'PASS' ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
                          {item.status}
                        </div>
                        {item.is_overridden && (
                          <span className="text-[10px] text-amber-400 flex items-center gap-1 font-semibold">
                            <ShieldCheck size={12} /> Overridden
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-4 px-4 text-white capitalize">
                      {item.defect_type === 'good' ? '-' : (item.defect_type ?? '').replaceAll('_', ' ')}
                    </td>
                    <td className="py-4 px-4">
                      <span className={`text-sm ${
                        item.severity_level === 'Critical' ? 'text-red-500 font-bold' :
                        item.severity_level === 'High' ? 'text-orange-400 font-bold' :
                        item.severity_level === 'Medium' ? 'text-amber-400 font-medium' : 'text-slate-400'
                      }`}>
                        {item.severity_level || '-'}
                      </span>
                    </td>
                    <td className="py-4 px-4 text-right text-slate-300 font-mono">
                      {typeof item.confidence === 'number' ? `${(item.confidence * 100).toFixed(1)}%` : '-'}
                    </td>
                    {(userRole === 'supervisor' || userRole === 'admin') && (
                      <td className="py-4 px-4 text-center">
                        <button
                          onClick={() => handleOpenOverride(item)}
                          className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 rounded-lg border border-slate-700 transition flex items-center gap-1 mx-auto"
                        >
                          <Edit3 size={13} /> Override
                        </button>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Supervisor Override Modal */}
      {selectedInspection && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="glass-panel p-6 max-w-lg w-full space-y-5 rounded-2xl relative border border-slate-700">
            <button
              onClick={() => setSelectedInspection(null)}
              className="absolute right-4 top-4 text-slate-400 hover:text-white"
            >
              <X size={20} />
            </button>

            <h3 className="text-xl font-bold text-white flex items-center gap-2">
              <ShieldCheck className="text-amber-400" size={22} />
              Supervisor Override — Inspection #{selectedInspection.id}
            </h3>

            <div className="space-y-4">
              <div>
                <label className="text-xs text-slate-400 uppercase font-semibold">Current System Decision</label>
                <div className="text-sm text-slate-200 mt-1 font-mono">
                  {selectedInspection.status} (Defect: {selectedInspection.defect_type})
                </div>
              </div>

              <div>
                <label className="text-xs text-slate-400 uppercase font-semibold">New Manual Decision</label>
                <div className="flex gap-4 mt-2">
                  <button
                    type="button"
                    onClick={() => setOverrideStatus('PASS')}
                    className={`flex-1 py-2.5 rounded-xl font-bold text-sm transition border ${
                      overrideStatus === 'PASS'
                        ? 'bg-emerald-600 text-white border-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.3)]'
                        : 'bg-slate-900/60 text-slate-400 border-slate-800'
                    }`}
                  >
                    FORCE PASS
                  </button>
                  <button
                    type="button"
                    onClick={() => setOverrideStatus('FAIL')}
                    className={`flex-1 py-2.5 rounded-xl font-bold text-sm transition border ${
                      overrideStatus === 'FAIL'
                        ? 'bg-red-600 text-white border-red-500 shadow-[0_0_15px_rgba(239,68,68,0.3)]'
                        : 'bg-slate-900/60 text-slate-400 border-slate-800'
                    }`}
                  >
                    FORCE FAIL
                  </button>
                </div>
              </div>

              <div>
                <label className="text-xs text-slate-400 uppercase font-semibold">Override Reason / Justification</label>
                <textarea
                  rows={3}
                  value={overrideReason}
                  onChange={(e) => setOverrideReason(e.target.value)}
                  placeholder="Explain why this decision is being manually overridden..."
                  className="w-full mt-1.5 p-3 bg-slate-900/80 border border-slate-700 rounded-xl text-white text-sm focus:outline-none focus:border-amber-500"
                />
              </div>
            </div>

            <div className="flex justify-end gap-3 pt-2">
              <button
                onClick={() => setSelectedInspection(null)}
                className="px-4 py-2 bg-slate-800 text-slate-300 rounded-xl text-sm"
              >
                Cancel
              </button>
              <button
                onClick={handleSubmitOverride}
                disabled={submittingOverride}
                className="px-5 py-2 bg-amber-600 hover:bg-amber-500 text-white font-medium rounded-xl text-sm transition shadow-[0_0_15px_rgba(217,119,6,0.4)]"
              >
                {submittingOverride ? 'Saving...' : 'Confirm Override'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
