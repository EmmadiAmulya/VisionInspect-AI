"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import api from '@/lib/api';
import { Lock, User } from 'lucide-react';
import axios from 'axios';

interface ValidationIssue {
  msg?: string;
}

function errorDetailMessage(err: unknown, fallback: string): string {
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

export default function LoginPage() {
  const router = useRouter();
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({ username: '', password: '', email: '' });
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const token = typeof window !== 'undefined' && (localStorage.getItem('token') || localStorage.getItem('access_token'));
    if (token) router.replace('/');
  }, [router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      if (isLogin) {
        const params = new URLSearchParams();
        params.append('username', formData.username);
        params.append('password', formData.password);
        
        const res = await api.post('/auth/login', params.toString(), {
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
        });
        
        localStorage.setItem('token', res.data.access_token);
        localStorage.setItem('access_token', res.data.access_token);
        
        // Fetch user role using the token just received
        try {
          const meRes = await api.get('/auth/me', {
            headers: { Authorization: `Bearer ${res.data.access_token}` }
          });
          localStorage.setItem('role', meRes.data.role);
          localStorage.setItem('username', meRes.data.username);
        } catch {
          localStorage.removeItem('token');
          localStorage.removeItem('access_token');
          throw new Error('Failed to fetch user profile');
        }
        
        router.replace('/');
      } else {
        if (formData.password.length < 6) {
          setError('Password must be at least 6 characters.');
          setLoading(false);
          return;
        }
        await api.post('/auth/register', {
          username: formData.username,
          email: formData.email,
          password: formData.password
        });
        setIsLogin(true);
        setSuccess("Registration successful. Please login.");
      }
    } catch (err: unknown) {
      setError(errorDetailMessage(err, 'Authentication failed'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0f172a] fixed inset-0 z-50 p-4">
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-blue-600/20 blur-[120px]"></div>
        <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-sky-500/20 blur-[120px]"></div>
      </div>

      <div className="glass-card w-full max-w-md p-8 md:p-10 rounded-3xl relative z-10 border border-slate-700/50">
        <div className="w-16 h-16 mx-auto rounded-full bg-gradient-to-tr from-blue-600 to-sky-400 flex items-center justify-center mb-6 shadow-[0_0_30px_rgba(37,99,235,0.4)]">
          <span className="text-white font-bold text-2xl">V</span>
        </div>
        
        <h1 className="text-3xl font-bold text-white text-center tracking-tight mb-2">VisionInspect AI</h1>
        <p className="text-slate-400 text-center mb-8">{isLogin ? 'Sign in to your account' : 'Create a new account'}</p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1">
            <label className="text-sm font-medium text-slate-300 ml-1">Username</label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <User size={18} className="text-slate-500" />
              </div>
              <input
                type="text"
                required
                value={formData.username}
                onChange={(e) => setFormData({...formData, username: e.target.value})}
                className="w-full bg-slate-900/50 border border-slate-700 text-white rounded-xl pl-11 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-all placeholder-slate-500"
                placeholder="Enter username"
              />
            </div>
          </div>

          {!isLogin && (
            <div className="space-y-1">
              <label className="text-sm font-medium text-slate-300 ml-1">Email</label>
              <div className="relative">
                <input
                  type="email"
                  required
                  value={formData.email}
                  onChange={(e) => setFormData({...formData, email: e.target.value})}
                  className="w-full bg-slate-900/50 border border-slate-700 text-white rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-all placeholder-slate-500"
                  placeholder="Enter email address"
                />
              </div>
            </div>
          )}

          <div className="space-y-1">
            <label className="text-sm font-medium text-slate-300 ml-1">Password</label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Lock size={18} className="text-slate-500" />
              </div>
              <input
                type="password"
                required
                value={formData.password}
                onChange={(e) => setFormData({...formData, password: e.target.value})}
                className="w-full bg-slate-900/50 border border-slate-700 text-white rounded-xl pl-11 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-all placeholder-slate-500"
                placeholder="Enter password"
              />
            </div>
          </div>

          {error && (
            <div className="text-sm p-3 rounded-lg bg-red-500/10 text-red-400 border border-red-500/20">
              {error}
            </div>
          )}

          {success && (
            <div className="text-sm p-3 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              {success}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3.5 rounded-xl transition-all shadow-[0_0_20px_rgba(37,99,235,0.25)] hover:shadow-[0_0_25px_rgba(37,99,235,0.4)] mt-4 disabled:opacity-70 disabled:cursor-wait"
          >
            {loading ? 'Processing...' : (isLogin ? 'Sign In' : 'Create Account')}
          </button>
        </form>

        <div className="mt-6 text-center">
          <button 
            type="button"
            onClick={() => {setIsLogin(!isLogin); setError(null);}}
            className="text-sm text-slate-400 hover:text-white transition-colors"
          >
            {isLogin ? "Don't have an account? Sign up" : "Already have an account? Sign in"}
          </button>
        </div>
      </div>
    </div>
  );
}
