"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import api from '@/lib/api';
import axios from 'axios';
import {
  XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
  Area, AreaChart
} from 'recharts';
import { Activity, AlertTriangle, CheckCircle, Package } from 'lucide-react';

interface ValidationIssue {
  msg?: string;
}

interface TrendPoint {
  date: string;
  passed: number;
  failed: number;
}

export default function Dashboard() {
  const router = useRouter();
  interface DashboardStats {
    total_inspections?: number;
    passed?: number;
    failed?: number;
    pending?: number;
    defect_count?: number;
    defect_rate?: number;
    defect_distribution?: Record<string, number>;
  }
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [trends, setTrends] = useState<TrendPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    const token = localStorage.getItem('token') || localStorage.getItem('access_token');
    if (!token) {
      router.push('/login');
      return;
    }

    const fetchDashboardData = async () => {
      try {
        const [analyticsRes, trendsRes] = await Promise.all([
          api.get('/inspections/analytics'),
          api.get('/inspections/trends')
        ]);
        
        setStats(analyticsRes.data);
        const raw = trendsRes.data as { trends?: unknown } | unknown[] | undefined;
        const t: TrendPoint[] = Array.isArray(raw)
          ? (raw as TrendPoint[])
          : Array.isArray((raw as { trends?: unknown } | undefined)?.trends)
            ? ((raw as { trends: TrendPoint[] }).trends)
            : [];
        setTrends(t);
      } catch (err: unknown) {
        console.error("Failed to fetch dashboard data", err);
        const detail = axios.isAxiosError(err)
          ? (err.response?.data as { detail?: unknown } | undefined)?.detail
          : undefined;
        const msg = Array.isArray(detail)
          ? detail.map((d) => (typeof d === 'object' && d !== null && 'msg' in d ? String((d as ValidationIssue).msg) : '')).filter(Boolean).join(', ')
          : typeof detail === 'string'
            ? detail
            : err instanceof Error
              ? err.message
              : 'Failed to load dashboard';
        setLoadError(msg || 'Failed to load dashboard');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, [router]);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="glass-card p-8 rounded-2xl text-center text-red-400">
        {loadError}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="glass-card p-6 rounded-2xl flex items-center gap-4">
          <div className="w-14 h-14 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400">
            <Package size={28} />
          </div>
          <div>
            <p className="text-sm text-slate-400 font-medium">Total Inspections</p>
            <h3 className="text-3xl font-bold text-white mt-1">{stats?.total_inspections || 0}</h3>
          </div>
        </div>

        <div className="glass-card p-6 rounded-2xl flex items-center gap-4">
          <div className="w-14 h-14 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-400">
            <CheckCircle size={28} />
          </div>
          <div>
            <p className="text-sm text-slate-400 font-medium">Passed</p>
            <h3 className="text-3xl font-bold text-white mt-1">{stats?.passed || 0}</h3>
          </div>
        </div>

        <div className="glass-card p-6 rounded-2xl flex items-center gap-4">
          <div className="w-14 h-14 rounded-full bg-red-500/20 flex items-center justify-center text-red-400">
            <AlertTriangle size={28} />
          </div>
          <div>
            <p className="text-sm text-slate-400 font-medium">Failed</p>
            <h3 className="text-3xl font-bold text-white mt-1">{stats?.failed || 0}</h3>
          </div>
        </div>

        <div className="glass-card p-6 rounded-2xl flex items-center gap-4">
          <div className="w-14 h-14 rounded-full bg-amber-500/20 flex items-center justify-center text-amber-400">
            <Activity size={28} />
          </div>
          <div>
            <p className="text-sm text-slate-400 font-medium">Defect Rate</p>
            <h3 className="text-3xl font-bold text-white mt-1">{typeof stats?.defect_rate === 'number' ? stats.defect_rate.toFixed(2) : 0}%</h3>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Trend Chart */}
        <div className="glass-card p-6 rounded-2xl lg:col-span-2">
          <h3 className="text-lg font-bold text-white mb-6">Inspection Trends</h3>
          <div className="h-72">
            {trends.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trends}>
                  <defs>
                    <linearGradient id="colorPass" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorFail" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                  <XAxis dataKey="date" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                  <RechartsTooltip 
                    contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', color: '#fff' }}
                    itemStyle={{ color: '#fff' }}
                  />
                  <Area type="monotone" dataKey="passed" stroke="#10b981" strokeWidth={3} fillOpacity={1} fill="url(#colorPass)" />
                  <Area type="monotone" dataKey="failed" stroke="#ef4444" strokeWidth={3} fillOpacity={1} fill="url(#colorFail)" />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-slate-500">No trend data available</div>
            )}
          </div>
        </div>

        {/* Defect Distribution */}
        <div className="glass-card p-6 rounded-2xl">
          <h3 className="text-lg font-bold text-white mb-6">Defect Distribution</h3>
          <div className="space-y-4">
            {stats?.defect_distribution && Object.entries(stats.defect_distribution).map(([type, count]) => {
              const numCount = Number(count) || 0;
              const defectTotal = Number(stats.defect_count) || numCount || 1;
              const raw = Math.round((numCount / defectTotal) * 100) || 0;
              const percentage = Math.min(100, Math.max(0, raw));
              return (
                <div key={type}>
                  <div className="flex justify-between text-sm mb-1 text-slate-300">
                    <span className="capitalize">{type.replaceAll('_', ' ')}</span>
                    <span className="font-bold">{numCount} ({percentage}%)</span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-2.5">
                    <div 
                      className="bg-blue-500 h-2.5 rounded-full shadow-[0_0_10px_rgba(59,130,246,0.8)]" 
                      style={{ width: `${percentage}%` }}
                    ></div>
                  </div>
                </div>
              )
            })}
            
            {(!stats?.defect_distribution || Object.keys(stats.defect_distribution).length === 0) && (
               <div className="text-center text-slate-500 py-8">No defects recorded yet.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
