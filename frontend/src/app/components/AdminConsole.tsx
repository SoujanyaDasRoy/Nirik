import React, { useState, useEffect } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Shield, Activity, Database, Server, Info } from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://projectmantra-nirikshon-backend.hf.space";

interface HealthStatus {
  status: string;
  model_loaded: boolean;
  database: string;
  uptime_seconds: number;
  version: string;
}

export default function AdminConsole() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`, { credentials: "include" });
        if (res.ok) {
          const data = await res.json();
          setHealth(data);
        }
      } catch { /* silent */ }
      finally { setLoading(false); }
    };
    fetchHealth();
    const iv = setInterval(fetchHealth, 30000);
    return () => clearInterval(iv);
  }, []);

  const formatUptime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h}h ${m}m ${s}s`;
  };

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6 animate-fadein">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">System Administration</h2>
          <p className="text-sm text-muted-foreground mt-1">Backend health, model status, and runtime diagnostics.</p>
        </div>
        <Badge variant="outline" className="h-8 uppercase font-bold text-xs">Admin Access</Badge>
      </div>

      {/* Academic prototype notice */}
      <div className="flex items-start gap-3 p-4 border border-blue-500/20 bg-blue-500/5 rounded-xl text-xs">
        <Info className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" strokeWidth={1.5} />
        <div>
          <p className="font-bold text-blue-700 dark:text-blue-400">Academic Prototype</p>
          <p className="text-muted-foreground mt-0.5 leading-relaxed">
            This is a research prototype for a university final year project. User management, API key generation,
            and multi-tenant RBAC are not implemented and are out of scope for this system.
          </p>
        </div>
      </div>

      {/* System Health Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Card className="shadow-none border border-border bg-card">
          <CardContent className="p-5 space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                <Server className="w-4 h-4 text-primary" strokeWidth={1.5} />
              </div>
              <h4 className="font-bold text-sm">Flask API Server</h4>
            </div>
            {loading ? (
              <div className="space-y-2">
                <div className="h-4 w-3/4 bg-muted/50 animate-pulse rounded" />
                <div className="h-4 w-1/2 bg-muted/50 animate-pulse rounded" />
              </div>
            ) : health ? (
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Status</span>
                  <span className="font-bold text-emerald-500 capitalize">{health.status}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Version</span>
                  <span className="font-mono font-semibold">{health.version || "1.0.0"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Uptime</span>
                  <span className="font-mono font-semibold">{formatUptime(health.uptime_seconds || 0)}</span>
                </div>
              </div>
            ) : (
              <p className="text-xs text-destructive">Could not reach backend server.</p>
            )}
          </CardContent>
        </Card>

        <Card className="shadow-none border border-border bg-card">
          <CardContent className="p-5 space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                <Activity className="w-4 h-4 text-primary" strokeWidth={1.5} />
              </div>
              <h4 className="font-bold text-sm">AI Model (DenseNet-121)</h4>
            </div>
            {loading ? (
              <div className="space-y-2">
                <div className="h-4 w-3/4 bg-muted/50 animate-pulse rounded" />
                <div className="h-4 w-1/2 bg-muted/50 animate-pulse rounded" />
              </div>
            ) : health ? (
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Model Loaded</span>
                  <span className={`font-bold ${health.model_loaded ? "text-emerald-500" : "text-destructive"}`}>
                    {health.model_loaded ? "✓ Loaded" : "✗ Not Loaded"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Architecture</span>
                  <span className="font-mono font-semibold">DenseNet-121</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Input Shape</span>
                  <span className="font-mono font-semibold">224 × 224 × 3</span>
                </div>
              </div>
            ) : (
              <p className="text-xs text-destructive">Status unavailable.</p>
            )}
          </CardContent>
        </Card>

        <Card className="shadow-none border border-border bg-card">
          <CardContent className="p-5 space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                <Database className="w-4 h-4 text-primary" strokeWidth={1.5} />
              </div>
              <h4 className="font-bold text-sm">Database (SQLite)</h4>
            </div>
            {loading ? (
              <div className="space-y-2">
                <div className="h-4 w-3/4 bg-muted/50 animate-pulse rounded" />
              </div>
            ) : health ? (
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Connection</span>
                  <span className={`font-bold ${health.database === "ok" ? "text-emerald-500" : "text-amber-500"}`}>
                    {health.database === "ok" ? "✓ Connected" : health.database}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Engine</span>
                  <span className="font-mono font-semibold">SQLite 3 (local)</span>
                </div>
              </div>
            ) : (
              <p className="text-xs text-destructive">Status unavailable.</p>
            )}
          </CardContent>
        </Card>

        <Card className="shadow-none border border-border bg-card">
          <CardContent className="p-5 space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                <Shield className="w-4 h-4 text-primary" strokeWidth={1.5} />
              </div>
              <h4 className="font-bold text-sm">Authentication</h4>
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Method</span>
                <span className="font-semibold">Session Cookie + CSRF</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Roles Available</span>
                <span className="font-semibold">Admin, Reviewer</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">RBAC</span>
                <span className="font-semibold text-emerald-500">Enforced</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
