"use client";

import { useEffect, useState } from 'react';
import { usePathname } from 'next/navigation';
import { Bell, User } from 'lucide-react';

const TITLES: Record<string, string> = {
  '/': 'Dashboard Overview',
  '/inspect': 'Inspection Suite',
  '/history': 'Inspection History',
  '/settings': 'Quality Settings',
  '/login': 'Login',
};

export default function TopNav() {
  const pathname = usePathname();
  // Static initial value so SSR/prerender matches first client render.
  // Client-only username/role syncs after mount (see effect below).
  const [username, setUsername] = useState<string>('Inspector');
  const [role, setRole] = useState<string>('');

  useEffect(() => {
    const readUsername = () =>
      localStorage.getItem('username') || 'Inspector';
    const readRole = () => localStorage.getItem('role') || '';
    // Defer so no synchronous setState in effect body.
    queueMicrotask(() => {
      setUsername(readUsername());
      setRole(readRole());
    });
    const sync = () => {
      setUsername(readUsername());
      setRole(readRole());
    };
    window.addEventListener('storage', sync);
    window.addEventListener('focus', sync);
    return () => {
      window.removeEventListener('storage', sync);
      window.removeEventListener('focus', sync);
    };
  }, []);

  // Don't show topnav on login page
  if (pathname === '/login') return null;

  return (
    <header className="h-20 glass-panel fixed top-0 right-0 left-0 md:left-64 z-10 flex items-center justify-between px-8">
      <div>
        <h2 className="text-xl font-bold text-white capitalize tracking-wide">
          {TITLES[pathname ?? '/'] ?? (pathname === '/' ? 'Dashboard Overview' : (pathname ?? '').slice(1).replace(/\//g, ' / '))}
        </h2>
        <p className="text-sm text-slate-400">Manufacturing Defect Detection System</p>
      </div>

      <div className="flex items-center gap-5">
        <button type="button" aria-label="Notifications" className="w-10 h-10 rounded-full bg-slate-800/50 flex items-center justify-center text-slate-300 hover:text-white hover:bg-slate-700 transition-colors relative">
          <Bell size={20} />
          <span className="absolute top-2 right-2 w-2 h-2 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.8)]"></span>
        </button>
        
        <div className="flex items-center gap-3 bg-slate-800/50 pl-2 pr-4 py-1.5 rounded-full border border-[rgba(255,255,255,0.05)]">
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-600 to-sky-400 flex items-center justify-center">
            <User size={16} className="text-white" />
          </div>
          <span suppressHydrationWarning className="text-sm font-medium text-slate-300">{username}{role ? ` (${role})` : ''}</span>
        </div>
      </div>
    </header>
  );
}
