"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Activity, Lock, User, CheckCircle, AlertCircle, ArrowUpRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username || !password) {
      setError("Please fill in all fields.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE}/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username, password }),
        credentials: "include"
      });

      const data = await response.json();

      if (response.ok && data.success) {
        // Redirect to Workbench
        router.push("/diagnose");
      } else {
        setError(data.error || "Invalid username or password.");
      }
    } catch (err) {
      console.error("Login failed:", err);
      setError("Failed to connect to authentication server. Ensure Flask API is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col font-sans transition-colors duration-200">
      <div className="flex-1 flex flex-col justify-center items-center p-6 w-full max-w-md mx-auto space-y-6">

        {/* Brand Header */}
        <div className="text-center space-y-2">
          <div className="inline-flex w-12 h-12 rounded-full bg-primary/5 items-center justify-center text-primary mb-2">
            <Activity className="w-6 h-6" strokeWidth={1.5} />
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-foreground font-sans">
            Nirikshon Workstation
          </h2>
          <p className="text-xs text-muted-foreground max-w-xs mx-auto">
            Clinical AI Screening Suite for Pulmonary Tuberculosis
          </p>
        </div>

        {/* Login Card */}
        <Card className="border border-border bg-card rounded-xl shadow-md overflow-hidden w-full">
          <CardContent className="p-6 space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-foreground">
              Clinical Portal Ingress
            </h3>
            <Separator />

            {error && (
              <div className="p-3 border border-destructive/20 bg-destructive/5 rounded-xl flex items-start gap-2.5 text-xs text-destructive">
                <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}

            <form onSubmit={handleLogin} className="space-y-4">
              {/* Username */}
              <div className="space-y-1.5">
                <label className="text-muted-foreground font-semibold uppercase text-[10px]">
                  Credentialed Username
                </label>
                <div className="relative">
                  <User className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder="e.g. reviewer"
                    className="w-full pl-11 pr-4 py-2 border border-border rounded-full bg-card text-foreground focus:ring-1 focus:ring-primary outline-none h-10 text-xs"
                    disabled={loading}
                  />
                </div>
              </div>

              {/* Password */}
              <div className="space-y-1.5">
                <label className="text-muted-foreground font-semibold uppercase text-[10px]">
                  Account Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full pl-11 pr-4 py-2 border border-border rounded-full bg-card text-foreground focus:ring-1 focus:ring-primary outline-none h-10 text-xs"
                    disabled={loading}
                  />
                </div>
              </div>

              {/* Submit */}
              <Button
                type="submit"
                disabled={loading}
                className="w-full h-10 text-xs font-semibold rounded-full cursor-pointer mt-2"
              >
                {loading ? "Authenticating..." : "Sign In to Workstation"}
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Demo Credentials Alert Info */}
        <div className="p-4 border border-border/80 bg-muted/20 rounded-xl flex flex-col gap-1 text-[11px] leading-relaxed text-muted-foreground w-full">
          <p className="font-semibold text-foreground flex items-center gap-1.5">
            <CheckCircle className="w-3.5 h-3.5 text-emerald-500" /> Default Seed Accounts
          </p>
          <p className="text-muted-foreground mt-0.5">
            <strong>Reviewer:</strong> <span className="font-mono bg-muted/40 px-1 py-0.5 rounded">reviewer</span> / <span className="font-mono bg-muted/40 px-1 py-0.5 rounded">password123</span>
          </p>
          <p className="text-muted-foreground">
            <strong>Administrator:</strong> <span className="font-mono bg-muted/40 px-1 py-0.5 rounded">admin</span> / <span className="font-mono bg-muted/40 px-1 py-0.5 rounded">password123</span>
          </p>
        </div>

        {/* Return to Landing portal */}
        <div className="text-center">
          <button
            onClick={() => router.push("/")}
            className="text-[10px] text-muted-foreground hover:text-primary font-bold hover:underline cursor-pointer"
          >
            ← Back to Abstract Overview
          </button>
        </div>

      </div>

      {/* ── FOOTER ── */}
      <footer className="bg-card text-muted-foreground py-8 px-6 flex-shrink-0 text-xs border-t border-border transition-colors duration-200">
        <div className="w-full max-w-6xl mx-auto flex flex-col sm:flex-row justify-between items-center gap-4 text-muted-foreground font-sans text-xs">
          <p className="text-center sm:text-left">
            &copy; {new Date().getFullYear()} Nirikhshon. Academic Final Year Project Prototype.
          </p>
          <div className="flex gap-4">
            <span className="text-muted-foreground/80">Built with React, Next.js, and Flask</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
