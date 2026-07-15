"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Activity, CheckCircle, AlertCircle } from "lucide-react";
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
        const errMessage = typeof data.error === 'string' 
          ? data.error 
          : (data.error?.message || "Invalid credentials.");
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
    <div className="min-h-screen bg-[#090909] flex flex-col font-sans overflow-hidden relative selection:bg-[#0099ff] selection:text-[#ffffff]">
      
      {/* Framer-style Top Nav */}
      <nav className="h-[56px] bg-[#090909] flex items-center justify-between px-6 z-20 sticky top-0 border-b border-[#262626]">
        <div className="flex items-center gap-2 cursor-pointer" onClick={() => router.push("/")}>
          <Activity className="w-5 h-5 text-[#ffffff]" />
          <span className="text-[#ffffff] text-[15px] font-medium tracking-[-0.15px]">Nirikshon</span>
        </div>
        <button
          onClick={() => router.push("/")}
          className="text-[#999999] text-[13px] font-medium hover:text-[#ffffff] transition-colors"
        >
          Return to Overview
        </button>
      </nav>

      {/* Main Content */}
      <div className="flex-1 flex flex-col justify-center items-center p-6 w-full max-w-lg mx-auto z-10 relative">
        
        {/* Framer-style Display Typography */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="text-center space-y-6 w-full mb-12"
        >
          <h2 className="text-[42px] sm:text-[62px] font-medium tracking-[-2px] sm:tracking-[-3.1px] text-[#ffffff] leading-[1.00]">
            Clinical Ingress
          </h2>
          <p className="text-[24px] text-[#999999] leading-[1.30] tracking-[-0.01px] max-w-sm mx-auto">
            Secure authentication for the Nirikshon AI diagnostic workspace.
          </p>
        </motion.div>

        {/* Form Container (Charcoal Surface) */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
          className="w-full"
        >
          <div className="bg-[#141414] rounded-[20px] p-8 w-full">
            <AnimatePresence mode="wait">
              {error && (
                <motion.div
                  initial={{ opacity: 0, height: 0, marginBottom: 0 }}
                  animate={{ opacity: 1, height: "auto", marginBottom: 20 }}
                  exit={{ opacity: 0, height: 0, marginBottom: 0 }}
                  className="overflow-hidden"
                >
                  <div className="p-4 bg-[#090909] border border-[#ff5577]/30 rounded-[10px] flex items-start gap-3 text-[14px] text-[#ff5577] font-medium tracking-[-0.14px]">
                    <AlertCircle className="w-5 h-5 flex-shrink-0" />
                    <span className="mt-[2px]">{error}</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <form onSubmit={handleLogin} className="space-y-6">
              <div className="space-y-3">
                <label className="text-[#999999] text-[15px] font-medium tracking-[-0.15px]">
                  Credentialed Username
                </label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="e.g. reviewer"
                  className="w-full px-[14px] py-[10px] bg-[#090909] text-[#ffffff] text-[15px] rounded-[10px] border border-[#262626] outline-none transition-shadow focus:shadow-[0_0_0_1px_rgba(0,153,255,0.25)] focus:border-[#0099ff] placeholder:text-[#666666]"
                  disabled={loading}
                />
              </div>

              <div className="space-y-3">
                <label className="text-[#999999] text-[15px] font-medium tracking-[-0.15px]">
                  Account Password
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full px-[14px] py-[10px] bg-[#090909] text-[#ffffff] text-[15px] rounded-[10px] border border-[#262626] outline-none transition-shadow focus:shadow-[0_0_0_1px_rgba(0,153,255,0.25)] focus:border-[#0099ff] placeholder:text-[#666666]"
                  disabled={loading}
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full py-[15px] px-[15px] mt-4 bg-[#ffffff] text-[#000000] rounded-full text-[15px] font-semibold tracking-[-0.14px] hover:scale-[1.02] active:scale-100 transition-transform cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? "Authenticating..." : "Sign In to Workstation"}
              </button>
            </form>
          </div>
        </motion.div>

        {/* Framer-style Seed Accounts Box */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="w-full mt-8"
        >
          <div className="bg-[#090909] border border-[#262626] rounded-[15px] p-6 space-y-4">
            <p className="text-[13px] font-medium text-[#999999] tracking-[-0.13px] flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-[#22c55e]" /> Default Seed Accounts
            </p>
            <div className="flex flex-col gap-3">
              <div className="flex justify-between items-center pb-3 border-b border-[#1a1a1a]">
                <span className="text-[14px] font-medium text-[#ffffff]">Reviewer</span>
                <span className="font-mono text-[#999999] text-[13px]">reviewer / password123</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-[14px] font-medium text-[#ffffff]">Admin</span>
                <span className="font-mono text-[#999999] text-[13px]">admin / password123</span>
              </div>
            </div>
          </div>
        </motion.div>

      </div>
    </div>
  );
}
