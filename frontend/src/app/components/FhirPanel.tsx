"use client";

import { useState, useEffect, useRef } from "react";
import { 
  Search, 
  Server, 
  RefreshCw, 
  CheckCircle, 
  ShieldAlert, 
  Users, 
  Loader2 
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

interface FhirPatient {
  id: string; mrn: string; name: string; given: string; family: string;
  gender: string; birthDate: string; age: number;
  presenting_symptoms: string[]; referring_doctor: string;
  hospital: string; blood_group: string;
}

interface PacsNode {
  name: string; host: string; port: number; ae_title: string;
  status: "online" | "degraded" | "offline";
  ping_ms: number | null; last_seen: string;
}

interface FhirPanelProps {
  onPatientSelect?: (patient: FhirPatient) => void;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

export default function FhirPanel({ onPatientSelect }: FhirPanelProps) {
  const [search, setSearch]             = useState("");
  const [patients, setPatients]         = useState<FhirPatient[]>([]);
  const [isSearching, setIsSearching]   = useState(false);
  const [selectedPatient, setSelectedPatient] = useState<FhirPatient | null>(null);
  const [pacsNodes, setPacsNodes]       = useState<PacsNode[]>([]);
  const [pacsLoading, setPacsLoading]   = useState(false);
  const [subTab, setSubTab]             = useState<"ehr" | "pacs">("ehr");

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (search.trim().length < 2) { setPatients([]); return; }
    debounceRef.current = setTimeout(async () => {
      setIsSearching(true);
      try {
        const res  = await fetch(`${API_BASE}/fhir/patients?search=${encodeURIComponent(search)}`, {
          credentials: "include"
        });
        const data = await res.json();
        setPatients(data.entry || []);
      } catch { setPatients([]); }
      finally { setIsSearching(false); }
    }, 350);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [search]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchPacs = async () => {
    setPacsLoading(true);
    try {
      const res  = await fetch(`${API_BASE}/fhir/pacs/status`, {
        credentials: "include"
      });
      const data = await res.json();
      setPacsNodes(data.nodes || []);
    } catch { setPacsNodes([]); }
    finally { setPacsLoading(false); }
  };

  useEffect(() => {
    if (subTab === "pacs") fetchPacs();
    const iv = setInterval(() => { if (subTab === "pacs") fetchPacs(); }, 30000);
    return () => clearInterval(iv);
  }, [subTab]); // eslint-disable-line react-hooks/exhaustive-deps

  const statusRing = (s: string) =>
    s === "online" ? "bg-emerald-500" : s === "degraded" ? "bg-amber-400" : "bg-neutral-400";
  const statusText = (s: string) =>
    s === "online" ? "text-emerald-500" : s === "degraded" ? "text-amber-500" : "text-neutral-500";

  return (
    <div className="space-y-4">
      {/* Sub-tab toggle */}
      <div className="flex gap-2 pb-2 border-b border-border">
        <Button size="sm" variant={subTab === "ehr" ? "default" : "outline"} onClick={() => setSubTab("ehr")} className="h-10 text-xs">
          <Users className="w-3.5 h-3.5 mr-1.5" strokeWidth={1.5} /> EHR / Patient Search
        </Button>
        <Button size="sm" variant={subTab === "pacs" ? "default" : "outline"} onClick={() => setSubTab("pacs")} className="h-10 text-xs">
          <Server className="w-3.5 h-3.5 mr-1.5" strokeWidth={1.5} /> PACS Gateway
        </Button>
      </div>

      {/* ── EHR ── */}
      {subTab === "ehr" && (
        <div className="space-y-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" strokeWidth={1.5} />
            <input
              type="text" value={search} onChange={e => setSearch(e.target.value)}
              placeholder="Search by patient name or MRN…"
              className="w-full pl-10 pr-10 py-2.5 text-sm rounded-lg border border-border bg-card text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 transition h-11"
            />
            {isSearching && (
              <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 animate-spin h-4 w-4 text-primary" strokeWidth={1.5} />
            )}
          </div>

          {patients.length > 0 && (
            <div className="space-y-2 max-h-[260px] overflow-y-auto pr-1">
              {patients.map(p => (
                <button key={p.id} onClick={() => { setSelectedPatient(p); onPatientSelect?.(p); }}
                  className={`w-full text-left p-3 rounded-lg border transition text-sm cursor-pointer ${
                    selectedPatient?.id === p.id
                      ? "border-primary bg-primary/5"
                      : "border-border hover:border-primary/30 hover:bg-muted/40"
                  }`}>
                  <div className="flex justify-between">
                    <div>
                      <p className="font-bold text-foreground">{p.name}</p>
                      <p className="text-xs text-muted-foreground font-mono">{p.mrn} · {p.age}y · {p.gender}</p>
                      <p className="text-xs text-muted-foreground">{p.hospital}</p>
                    </div>
                    <p className="text-xs text-muted-foreground text-right max-w-[40%]">
                      {p.referring_doctor.split("(")[0].trim()}
                    </p>
                  </div>
                </button>
              ))}
            </div>
          )}

          {selectedPatient && (
            <Card className="border border-primary/30 bg-primary/5 rounded-lg shadow-none">
              <CardContent className="p-4 space-y-3">
                <div className="flex justify-between items-start">
                  <div>
                    <p className="font-extrabold text-sm text-foreground">{selectedPatient.name}</p>
                    <p className="text-xs text-muted-foreground font-mono">{selectedPatient.mrn} · Blood: {selectedPatient.blood_group}</p>
                  </div>
                  <Badge variant="outline" className="border-primary text-primary text-[10px] bg-card h-5 flex items-center gap-0.5">
                    <CheckCircle className="w-2.5 h-2.5 text-primary" strokeWidth={1.5} /> Auto-filled
                  </Badge>
                </div>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  {[
                    ["Date of Birth", `${selectedPatient.birthDate} (Age: ${selectedPatient.age})`],
                    ["Gender", selectedPatient.gender],
                    ["Hospital", selectedPatient.hospital],
                    ["Referring Doctor", selectedPatient.referring_doctor],
                  ].map(([k, v]) => (
                    <div key={k}>
                      <p className="text-[10px] text-muted-foreground uppercase font-medium mb-0.5">{k}</p>
                      <p className="font-semibold text-foreground">{v}</p>
                    </div>
                  ))}
                </div>
                <div>
                  <p className="text-[10px] text-muted-foreground uppercase font-medium mb-1.5">Presenting Symptoms</p>
                  <div className="flex flex-wrap gap-1.5">
                    {selectedPatient.presenting_symptoms.map((s, i) => (
                      <Badge key={i} variant="outline" className="text-[10px] font-normal border-amber-500/20 bg-amber-500/5 text-amber-600 dark:text-amber-500">
                        {s}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* ── PACS ── */}
      {subTab === "pacs" && (
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <div>
              <p className="font-bold text-sm">DICOM PACS Network</p>
              <p className="text-xs text-muted-foreground">Simulated nodes · refreshes every 30s</p>
            </div>
            <Button size="sm" variant="ghost" onClick={fetchPacs} disabled={pacsLoading} className="h-10 text-xs">
              {pacsLoading ? (
                <Loader2 className="animate-spin h-3.5 w-3.5 mr-1" strokeWidth={1.5} />
              ) : (
                <RefreshCw className="w-3.5 h-3.5 mr-1" strokeWidth={1.5} />
              )}
              Refresh
            </Button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {pacsNodes.map((node, i) => (
              <Card key={i} className={`border rounded-lg shadow-none bg-card ${
                node.status === "online" ? "border-emerald-500/20" :
                node.status === "degraded" ? "border-amber-400/20" :
                "border-neutral-500/20"
              }`}>
                <CardContent className="p-3">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="text-xs font-bold text-foreground">{node.name}</p>
                      <p className="text-[10px] text-muted-foreground font-mono">{node.host}:{node.port}</p>
                      <p className="text-[10px] text-muted-foreground">AE: {node.ae_title}</p>
                    </div>
                    <div className="text-right space-y-1">
                      <div className="flex items-center gap-1.5 justify-end">
                        <span className={`w-2 h-2 rounded-full ${statusRing(node.status)} ${node.status === "online" ? "animate-pulse" : ""}`} />
                        <span className={`text-[10px] font-bold capitalize ${statusText(node.status)}`}>{node.status}</span>
                      </div>
                      {node.ping_ms !== null && <p className="text-[10px] text-muted-foreground font-mono">{node.ping_ms}ms</p>}
                      <p className="text-[10px] text-muted-foreground font-mono">{node.last_seen}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          <div className="flex gap-4 text-[10px] text-muted-foreground pt-1">
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-emerald-500" />Online</span>
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-amber-400" />Degraded</span>
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-neutral-400" />Offline</span>
          </div>
        </div>
      )}
    </div>
  );
}
