"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Activity, Lock, User, CheckCircle, AlertCircle, ArrowUpRight } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://projectmantra-nirikshon-backend.hf.space";

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
        localStorage.setItem("nirikshon_user", JSON.stringify({ username: data.username, role: data.role }));
        router.push("/diagnose");
      } else {
        // Correctly handle the error from the new endpoint format, or fallback
        const errMessage = typeof data.error === 'string' 
          ? data.error 
          : (data.error?.message || "Invalid username or password.");
        setError(errMessage);
      }
    } catch (err) {
      console.error("Login failed:", err);
      setError("Failed to connect to authentication server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#090909] flex flex-col font-sans transition-colors duration-200 overflow-hidden relative">
      {/* Background glow effects */}
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-[#0099ff] opacity-[0.05] blur-[120px] rounded-full pointer-events-none" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-[#d44df0] opacity-[0.05] blur-[120px] rounded-full pointer-events-none" />
      
      <div className="flex-1 flex flex-col justify-center items-center p-6 w-full max-w-md mx-auto space-y-8 z-10 relative">
        
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="text-center space-y-4"
        >
          <div className="inline-flex w-14 h-14 rounded-full bg-[#141414] border border-[#262626] items-center justify-center text-[#ffffff] shadow-2xl">
            <Activity className="w-6 h-6" strokeWidth={1.5} />
          </div>
          <div>
            <h2 className="text-[28px] font-medium tracking-tight text-[#ffffff] leading-tight">
              Clinical Ingress
            </h2>
            <p className="text-[14px] text-[#999999] mt-2 max-w-xs mx-auto">
              Secure authentication for the Nirikshon AI workstation.
            </p>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
          className="w-full"
        >
          <div className="bg-[#141414] border border-[#262626] rounded-[24px] shadow-2xl p-8 backdrop-blur-xl">
            <AnimatePresence mode="wait">
              {error && (
                <motion.div
                  initial={{ opacity: 0, height: 0, marginBottom: 0 }}
                  animate={{ opacity: 1, height: "auto", marginBottom: 20 }}
                  exit={{ opacity: 0, height: 0, marginBottom: 0 }}
                  className="overflow-hidden"
                >
                  <div className="p-3 border border-red-500/20 bg-red-500/10 rounded-xl flex items-start gap-2.5 text-[13px] text-red-400">
                    <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                    <span>{error}</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <form onSubmit={handleLogin} className="space-y-5">
              <div className="space-y-2">
                <label className="text-[#999999] font-medium text-[13px]">
                  Credentialed Username
                </label>
                <div className="relative group">
                  <User className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-[#666666] group-focus-within:text-[#ffffff] transition-colors" />
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder="e.g. reviewer"
                    className="w-full pl-11 pr-4 py-3 border border-[#262626] rounded-xl bg-[#090909] text-[#ffffff] focus:border-[#ffffff]/50 focus:ring-1 focus:ring-[#ffffff]/50 outline-none h-12 text-[14px] transition-all placeholder:text-[#444444]"
                    disabled={loading}
                  />
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-[#999999] font-medium text-[13px]">
                  Account Password
                </label>
                <div className="relative group">
                  <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-[#666666] group-focus-within:text-[#ffffff] transition-colors" />
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full pl-11 pr-4 py-3 border border-[#262626] rounded-xl bg-[#090909] text-[#ffffff] focus:border-[#ffffff]/50 focus:ring-1 focus:ring-[#ffffff]/50 outline-none h-12 text-[14px] transition-all placeholder:text-[#444444]"
                    disabled={loading}
                  />
                </div>
              </div>

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                type="submit"
                disabled={loading}
                className="w-full h-12 text-[14px] font-semibold rounded-xl mt-4 bg-[#ffffff] text-[#000000] hover:bg-[#e0e0e0] transition-colors flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? "Authenticating..." : "Sign In to Workstation"}
                {!loading && <ArrowUpRight className="w-4 h-4" />}
              </motion.button>
            </form>
          </div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="w-full"
        >
          <div className="p-4 border border-[#262626] bg-[#141414]/50 rounded-xl flex flex-col gap-3 text-[13px] leading-relaxed text-[#999999]">
            <p className="font-semibold text-[#ffffff] flex items-center gap-1.5">
              <CheckCircle className="w-4 h-4 text-[#10b981]" /> Default Seed Accounts
            </p>
            <div className="flex justify-between items-center bg-[#090909] border border-[#262626] p-2.5 rounded-lg">
              <span>Reviewer</span>
              <span className="font-mono text-[#ffffff] tracking-wide">reviewer / password123</span>
            </div>
            <div className="flex justify-between items-center bg-[#090909] border border-[#262626] p-2.5 rounded-lg">
              <span>Admin</span>
              <span className="font-mono text-[#ffffff] tracking-wide">admin / password123</span>
            </div>
          </div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="text-center mt-4"
        >
          <button
            onClick={() => router.push("/")}
            className="text-[13px] text-[#666666] hover:text-[#ffffff] font-medium transition-colors cursor-pointer"
          >
            ← Return to Overview
          </button>
        </motion.div>

      </div>
    </div>
  );
}
