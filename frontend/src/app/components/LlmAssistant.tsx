"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Loader2, Sparkles, Image as ImageIcon } from "lucide-react";
import { AnalysisResult } from "../hooks/useFileUpload";

export default function LlmAssistant({ activeResult }: { activeResult: AnalysisResult | null }) {
  const [messages, setMessages] = useState<{ role: "assistant" | "user"; content: string }[]>([
    {
      role: "assistant",
      content: "Hello! I am your AI Diagnostic Assistant. You can ask me questions about the current radiograph, the model's confidence, or clinical observations."
    }
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const endOfMessagesRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // Check LLM status on mount
  useEffect(() => {
    const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://projectmantra-nirikshon-backend.hf.space";
    fetch(`${API_BASE}/chat/status`, { credentials: "include" })
      .then(r => r.json())
      .then(data => {
        if (!data.llm_available) {
          setMessages(prev => [...prev, {
            role: "assistant",
            content: "⚠️ AI Co-Pilot is running in degraded mode: GEMINI_API_KEY is not configured on the server. You will receive rule-based responses based on the model output. To enable full LLM responses, add the GEMINI_API_KEY to the Hugging Face Space secrets."
          }]);
        }
      })
      .catch(() => {}); // Silently ignore if status check fails
  }, []);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = input.trim();
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: userMessage }]);
    setIsTyping(true);

    try {
      const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://projectmantra-nirikshon-backend.hf.space";
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          message: userMessage,
          llm_context: {
            prediction: activeResult?.prediction || "Unknown",
            confidence: activeResult?.confidence || 0,
            threshold: activeResult?.threshold_used || 0.5,
            patientId: activeResult?.metadata?.patient_id || "Unknown Patient",
            isTb: activeResult?.is_tb || false
          }
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response");
      }

      const data = await response.json();
      setMessages(prev => [...prev, { role: "assistant", content: data.response || "No response generated." }]);
    } catch (error) {
      console.error("LLM Error:", error);
      const errMsg = error instanceof Error ? error.message : "Unknown error";
      setMessages(prev => [...prev, { role: "assistant", content: `⚠️ Could not reach the AI assistant (${errMsg}). Make sure the backend is running and GEMINI_API_KEY is set.` }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSend();
    }
  };

  const suggestions = [
    "Explain the prediction",
    "What is the confidence level?",
    "Check image quality",
  ];

  return (
    <div className="flex flex-col h-full bg-card/25 backdrop-blur-xl border border-border/50 rounded-2xl overflow-hidden shadow-lg animate-fadein">
      {/* Header */}
      <div className="flex items-center gap-2 p-4 border-b border-white/5 bg-black/20">
        <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
          <Sparkles className="w-4 h-4 text-primary" />
        </div>
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-foreground">AI Co-Pilot</h3>
          <p className="text-[10px] text-muted-foreground">Ask about {activeResult?.metadata?.patient_id || "this study"}</p>
        </div>
      </div>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} animate-fadein`}>
            <div className={`flex gap-3 max-w-[85%] ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}>
              <div className={`w-6 h-6 rounded-full flex-shrink-0 flex items-center justify-center mt-1 ${
                msg.role === "user" ? "bg-muted" : "bg-primary/20"
              }`}>
                {msg.role === "user" ? <User className="w-3 h-3 text-muted-foreground" /> : <Bot className="w-3 h-3 text-primary" />}
              </div>
              <div className={`p-3 rounded-[15px] text-xs leading-relaxed ${
                msg.role === "user" 
                  ? "bg-muted/40 text-foreground rounded-tr-sm" 
                  : "bg-black/40 border border-white/5 text-foreground rounded-tl-sm shadow-sm"
              }`}>
                {msg.content}
              </div>
            </div>
          </div>
        ))}

        {isTyping && (
          <div className="flex justify-start animate-fadein">
            <div className="flex gap-3 max-w-[85%] flex-row">
              <div className="w-6 h-6 rounded-full bg-primary/20 flex-shrink-0 flex items-center justify-center mt-1">
                <Bot className="w-3 h-3 text-primary" />
              </div>
              <div className="p-3 rounded-[15px] bg-black/40 border border-white/5 text-foreground rounded-tl-sm shadow-sm flex items-center gap-1.5">
                <Loader2 className="w-3 h-3 animate-spin text-primary" />
                <span className="text-[10px] text-muted-foreground animate-pulse">Analyzing context...</span>
              </div>
            </div>
          </div>
        )}
        <div ref={endOfMessagesRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 bg-black/20 border-t border-white/5">
        {/* Suggestion Chips */}
        {messages.length === 1 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {suggestions.map((s, idx) => (
              <button
                key={idx}
                onClick={() => setInput(s)}
                className="px-3 py-1.5 rounded-full border border-border/50 bg-muted/20 hover:bg-muted/40 text-[10px] text-muted-foreground transition-colors cursor-pointer"
              >
                {s}
              </button>
            ))}
          </div>
        )}

        <div className="flex items-center gap-2 relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question..."
            disabled={isTyping}
            className="w-full bg-muted/30 border border-border/80 rounded-full h-10 pl-4 pr-10 text-xs text-foreground outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all disabled:opacity-50"
          />
          <button 
            onClick={handleSend}
            disabled={!input.trim() || isTyping}
            className="absolute right-1 w-8 h-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground disabled:opacity-50 transition-transform active:scale-95 cursor-pointer"
          >
            <Send className="w-3.5 h-3.5" />
          </button>
        </div>
        <p className="text-[9px] text-center text-muted-foreground mt-2">AI can make mistakes. Always verify clinical findings.</p>
      </div>
    </div>
  );
}
