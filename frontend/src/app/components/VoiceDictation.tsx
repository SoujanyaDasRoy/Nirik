"use client";

import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

type Section = "indication" | "findings" | "impression" | "notes";

const SECTIONS: { id: Section; label: string; rows: number }[] = [
  { id: "indication", label: "Indication",        rows: 2 },
  { id: "findings",   label: "Findings",           rows: 4 },
  { id: "impression", label: "Impression",         rows: 2 },
  { id: "notes",      label: "Additional Notes",   rows: 2 },
];

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}
interface SpeechRecognitionErrorEvent extends Event { error: string; }
interface SpeechRecognitionInstance extends EventTarget {
  continuous: boolean; interimResults: boolean; lang: string;
  start(): void; stop(): void;
  onresult: ((e: SpeechRecognitionEvent) => void) | null;
  onerror:  ((e: SpeechRecognitionErrorEvent) => void) | null;
  onend:    (() => void) | null;
}
declare global {
  interface Window {
    SpeechRecognition: new () => SpeechRecognitionInstance;
    webkitSpeechRecognition: new () => SpeechRecognitionInstance;
  }
}

interface VoiceDictationProps {
  onNoteChange?: (note: string) => void;
}

export default function VoiceDictation({ onNoteChange }: VoiceDictationProps) {
  const [isListening, setIsListening]     = useState(false);
  const [activeSection, setActiveSection] = useState<Section>("findings");
  const [texts, setTexts]                 = useState<Record<Section, string>>({ indication: "", findings: "", impression: "", notes: "" });
  const [interimText, setInterimText]     = useState("");
  const [supported, setSupported]         = useState(true);
  const [error, setError]                 = useState<string | null>(null);

  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);

  const buildNote = (t: Record<Section, string>) =>
    SECTIONS.map(s => t[s.id] ? `${s.label}:\n${t[s.id].trim()}` : "").filter(Boolean).join("\n\n");

  const activeSectionRef = useRef(activeSection);
  const onNoteChangeRef  = useRef(onNoteChange);

  // Synchronize refs with current state / props
  useEffect(() => {
    activeSectionRef.current = activeSection;
  }, [activeSection]);

  useEffect(() => {
    onNoteChangeRef.current = onNoteChange;
  }, [onNoteChange]);

  useEffect(() => {
    const SR = typeof window !== "undefined" && (window.SpeechRecognition || window.webkitSpeechRecognition);
    if (!SR) { setSupported(false); return; }
    const r = new SR();
    r.continuous = true; r.interimResults = true; r.lang = "en-IN";
    r.onresult = (e: SpeechRecognitionEvent) => {
      let interim = "", final = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript;
        if (e.results[i].isFinal) final += t + " ";
        else interim += t;
      }
      if (final) {
        setTexts(prev => {
          const currentSection = activeSectionRef.current;
          const upd = { ...prev, [currentSection]: prev[currentSection] + final };
          onNoteChangeRef.current?.(buildNote(upd));
          return upd;
        });
      }
      setInterimText(interim);
    };
    r.onerror  = (e: SpeechRecognitionErrorEvent) => { setError(`Speech error: ${e.error}`); setIsListening(false); };
    r.onend    = () => { setIsListening(false); setInterimText(""); };
    recognitionRef.current = r;

    return () => {
      try {
        r.stop();
      } catch { /* silent */ }
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const toggleListening = () => {
    if (!recognitionRef.current) return;
    setError(null);
    if (isListening) { recognitionRef.current.stop(); setIsListening(false); }
    else {
      try { recognitionRef.current.start(); setIsListening(true); }
      catch { setError("Could not start microphone. Please allow microphone access."); }
    }
  };

  const clearAll = () => {
    const empty = { indication: "", findings: "", impression: "", notes: "" };
    setTexts(empty); onNoteChange?.("");
  };

  if (!supported) return (
    <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 text-sm text-amber-500">
      ⚠ Voice dictation requires Chrome or Edge browser (Web Speech API).
    </div>
  );

  return (
    <div className="space-y-4">
      {/* Section selector */}
      <div className="flex flex-wrap gap-2">
        {SECTIONS.map(s => (
          <Button key={s.id} size="sm"
            variant={activeSection === s.id ? "default" : "outline"}
            onClick={() => setActiveSection(s.id)}>
            {s.label}
          </Button>
        ))}
      </div>

      {/* Mic */}
      <div className="flex items-center gap-4">
        <Button
          onClick={toggleListening}
          variant={isListening ? "destructive" : "default"}
          className={isListening ? "relative shadow-lg shadow-destructive/30" : ""}
        >
          {isListening && <span className="absolute inset-0 rounded-md animate-ping bg-red-400 opacity-25" />}
          <svg className="w-4 h-4 mr-1.5 relative z-10" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 1a4 4 0 014 4v7a4 4 0 11-8 0V5a4 4 0 014-4zm0 2a2 2 0 00-2 2v7a2 2 0 004 0V5a2 2 0 00-2-2zm6.32 6.22A6.5 6.5 0 0119 12v1a7 7 0 01-6 6.93V22h-2v-2.07A7 7 0 015 13v-1a6.5 6.5 0 01.68-2.78l1.68 1.68A4.5 4.5 0 006.5 13v1a5.5 5.5 0 0011 0v-1a4.5 4.5 0 00-.86-2.1l1.68-1.68z" />
          </svg>
          <span className="relative z-10">
            {isListening ? "Stop Recording" : `Dictate ${SECTIONS.find(s => s.id === activeSection)?.label}`}
          </span>
        </Button>

        {isListening && (
          <div className="flex items-center gap-1.5">
            {[1, 2, 3, 4].map(i => (
              <span key={i} className="w-1 bg-red-500 rounded-full animate-bounce"
                style={{ height: `${8 + i * 4}px`, animationDelay: `${i * 80}ms` }} />
            ))}
            <Badge variant="destructive" className="text-[10px] ml-1">● Live</Badge>
          </div>
        )}
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {error}
        </div>
      )}

      <Separator />

      {/* Text sections */}
      {SECTIONS.map(sec => (
        <div key={sec.id} className={`space-y-1.5 transition-opacity ${activeSection === sec.id ? "opacity-100" : "opacity-60"}`}>
          <div className="flex justify-between items-center">
            <label className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              {sec.label}
              {activeSection === sec.id && isListening && (
                <span className="ml-2 text-red-500 animate-pulse text-[10px]">● Recording</span>
              )}
            </label>
            {texts[sec.id] && (
              <button onClick={() => {
                const upd = { ...texts, [sec.id]: "" };
                setTexts(upd); onNoteChange?.(buildNote(upd));
              }} className="text-[10px] text-destructive hover:underline">Clear</button>
            )}
          </div>
          <Textarea
            value={texts[sec.id] + (activeSection === sec.id && isListening ? interimText : "")}
            onChange={e => {
              const upd = { ...texts, [sec.id]: e.target.value };
              setTexts(upd); onNoteChange?.(buildNote(upd));
            }}
            onFocus={() => setActiveSection(sec.id)}
            placeholder={`Click to type or dictate ${sec.label.toLowerCase()}…`}
            rows={sec.rows}
            className="resize-none text-xs focus-visible:ring-primary"
          />
        </div>
      ))}

      <div className="flex justify-between items-center pt-2">
        <Button variant="ghost" size="sm" onClick={clearAll} className="text-muted-foreground text-xs">
          Clear All
        </Button>
        <span className="text-[10px] text-muted-foreground">
          {buildNote(texts).length} chars · included in PDF report
        </span>
      </div>
    </div>
  );
}
