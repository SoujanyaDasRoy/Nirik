"use client";

import { useState, useEffect } from "react";
import { 
  Search, 
  UserPlus, 
  Archive, 
  Edit, 
  ChevronRight, 
  Users, 
  ExternalLink,
  Info,
  Calendar,
  X,
  FileText
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

interface Patient {
  id: string;
  name: string;
  age: string;
  sex: string;
  notes: string;
  is_archived: number;
}

interface PatientsTabProps {
  onSelectStudy?: (record: any) => void;
}

export function PatientsTab({ onSelectStudy }: PatientsTabProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [patients, setPatients] = useState<Patient[]>([]);
  const [includeArchived, setIncludeArchived] = useState(false);
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [studies, setStudies] = useState<any[]>([]);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingStudies, setLoadingStudies] = useState(false);

  // Form Modals
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);

  // Form States
  const [formId, setFormId] = useState("");
  const [formName, setFormName] = useState("");
  const [formAge, setFormAge] = useState("");
  const [formSex, setFormSex] = useState("Male");
  const [formNotes, setFormNotes] = useState("");
  const [formError, setFormError] = useState("");

  const fetchPatients = async () => {
    setLoadingList(true);
    try {
      const url = `${API_BASE}/patients?search=${encodeURIComponent(searchQuery)}&include_archived=${includeArchived}`;
      const response = await fetch(url, { credentials: "include" });
      if (response.ok) {
        const data = await response.json();
        setPatients(data.patients || []);
      }
    } catch (err) {
      console.error("Error fetching patients list:", err);
    } finally {
      setLoadingList(false);
    }
  };

  const fetchStudiesHistory = async (patientId: string) => {
    setLoadingStudies(true);
    try {
      const response = await fetch(`${API_BASE}/patients/${patientId}/history`, { credentials: "include" });
      if (response.ok) {
        const data = await response.json();
        setStudies(data.records || []);
      }
    } catch (err) {
      console.error("Error loading patient study history:", err);
    } finally {
      setLoadingStudies(false);
    }
  };

  useEffect(() => {
    fetchPatients();
  }, [searchQuery, includeArchived]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (selectedPatient) {
      fetchStudiesHistory(selectedPatient.id);
    } else {
      setStudies([]);
    }
  }, [selectedPatient]);

  const handleCreatePatient = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formId || !formName) {
      setFormError("Patient ID and Name are required.");
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/patients`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: formId,
          name: formName,
          age: formAge || "N/A",
          sex: formSex,
          notes: formNotes
        }),
        credentials: "include"
      });

      const data = await response.json();
      if (response.ok) {
        setShowCreateModal(false);
        resetForm();
        fetchPatients();
      } else {
        setFormError(data.error || "Failed to create patient.");
      }
    } catch (err) {
      setFormError("Server connection failed.");
    }
  };

  const handleUpdatePatient = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedPatient) return;

    try {
      const response = await fetch(`${API_BASE}/patients/${selectedPatient.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: formName,
          age: formAge || "N/A",
          sex: formSex,
          notes: formNotes
        }),
        credentials: "include"
      });

      const data = await response.json();
      if (response.ok) {
        setShowEditModal(false);
        const updated = { ...selectedPatient, name: formName, age: formAge || "N/A", sex: formSex, notes: formNotes };
        setSelectedPatient(updated);
        fetchPatients();
      } else {
        setFormError(data.error || "Failed to update patient.");
      }
    } catch (err) {
      setFormError("Server connection failed.");
    }
  };

  const handleArchiveToggle = async (patient: Patient) => {
    const nextArchiveState = patient.is_archived === 0;
    try {
      const response = await fetch(`${API_BASE}/patients/${patient.id}/archive`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ archive: nextArchiveState }),
        credentials: "include"
      });
      if (response.ok) {
        if (selectedPatient?.id === patient.id) {
          setSelectedPatient(prev => prev ? { ...prev, is_archived: nextArchiveState ? 1 : 0 } : null);
        }
        fetchPatients();
      }
    } catch (err) {
      console.error("Failed to archive patient:", err);
    }
  };

  const openCreateModal = () => {
    resetForm();
    setFormError("");
    setShowCreateModal(true);
  };

  const openEditModal = () => {
    if (!selectedPatient) return;
    setFormId(selectedPatient.id);
    setFormName(selectedPatient.name);
    setFormAge(selectedPatient.age);
    setFormSex(selectedPatient.sex);
    setFormNotes(selectedPatient.notes);
    setFormError("");
    setShowEditModal(true);
  };

  const resetForm = () => {
    setFormId("");
    setFormName("");
    setFormAge("");
    setFormSex("Male");
    setFormNotes("");
    setFormError("");
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start relative max-w-5xl mx-auto">
      
      {/* LEFT COLUMN: PATIENT LISTING (Col span 5) */}
      <div className="lg:col-span-5 space-y-4">
        <div className="flex justify-between items-center gap-2">
          <h3 className="text-sm font-bold uppercase tracking-wider text-foreground">Cohort Directory</h3>
          <Button 
            onClick={openCreateModal}
            size="sm"
            className="rounded-full text-xs font-semibold h-8 gap-1 cursor-pointer"
          >
            <UserPlus className="w-3.5 h-3.5" /> New Patient
          </Button>
        </div>
        <Separator />

        {/* Search & Filters */}
        <div className="space-y-3">
          <div className="relative">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" strokeWidth={1.5} />
            <input
              type="text"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Search ID, Name..."
              className="w-full pl-10 pr-4 py-2 border border-border rounded-full bg-card text-foreground focus:ring-1 focus:ring-primary outline-none h-10 text-xs"
            />
          </div>
          
          <label className="flex items-center gap-2 text-xs text-muted-foreground select-none cursor-pointer">
            <input
              type="checkbox"
              checked={includeArchived}
              onChange={e => setIncludeArchived(e.target.checked)}
              className="rounded border-border text-primary focus:ring-primary h-3.5 w-3.5 cursor-pointer"
            />
            <span>Include Archived Patients</span>
          </label>
        </div>

        {/* Patients Grid */}
        <div className="space-y-2 max-h-[460px] overflow-y-auto pr-1 divide-y divide-border/40 scrollbar-none">
          {loadingList ? (
            <div className="p-12 text-center text-xs text-muted-foreground animate-pulse">
              Loading cohort registry...
            </div>
          ) : patients.length === 0 ? (
            <div className="p-8 text-center text-xs text-muted-foreground bg-muted/10 rounded-xl border border-dashed border-border">
              No patients found matching criteria.
            </div>
          ) : (
            patients.map(p => (
              <button
                key={p.id}
                onClick={() => setSelectedPatient(p)}
                className={`w-full text-left p-3.5 transition-all flex justify-between items-center cursor-pointer ${
                  selectedPatient?.id === p.id
                    ? "bg-primary/5 border-l-2 border-primary"
                    : "hover:bg-muted/30"
                }`}
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-1.5">
                    <span className="font-bold text-xs text-foreground font-sans">{p.name}</span>
                    {p.is_archived === 1 && (
                      <Badge variant="outline" className="text-[8px] px-1 py-0 rounded-md text-muted-foreground border-border/80 uppercase bg-muted/40">Archived</Badge>
                    )}
                  </div>
                  <span className="text-[10px] text-muted-foreground font-mono">
                    ID: {p.id} · {p.age}y · {p.sex}
                  </span>
                </div>
                <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />
              </button>
            ))
          )}
        </div>
      </div>

      {/* RIGHT COLUMN: PATIENT DETAILED VIEW (Col span 7) */}
      <div className="lg:col-span-7 space-y-6">
        {selectedPatient ? (
          <div className="space-y-6 animate-fadein">
            {/* Demographics Card */}
            <Card className="border border-border bg-card rounded-xl shadow-none">
              <CardContent className="p-6 space-y-4">
                <div className="flex justify-between items-start">
                  <div className="space-y-1">
                    <span className="text-[9px] font-extrabold uppercase text-muted-foreground tracking-widest">Selected Profile</span>
                    <h3 className="text-lg font-bold text-foreground font-sans">{selectedPatient.name}</h3>
                    <p className="text-[10px] text-muted-foreground font-mono">Database Case ID: {selectedPatient.id}</p>
                  </div>
                  
                  <div className="flex gap-2">
                    <Button
                      onClick={openEditModal}
                      variant="outline"
                      size="sm"
                      className="rounded-full text-xs h-8 px-3.5 gap-1.5 cursor-pointer"
                    >
                      <Edit className="w-3.5 h-3.5" /> Edit
                    </Button>
                    <Button
                      onClick={() => handleArchiveToggle(selectedPatient)}
                      variant="outline"
                      size="sm"
                      className={`rounded-full text-xs h-8 px-3.5 gap-1.5 cursor-pointer ${
                        selectedPatient.is_archived === 1 ? "text-primary border-primary/20 bg-primary/5" : "text-muted-foreground"
                      }`}
                    >
                      <Archive className="w-3.5 h-3.5" />
                      {selectedPatient.is_archived === 1 ? "Unarchive" : "Archive"}
                    </Button>
                  </div>
                </div>
                <Separator />

                {/* Demographics Grid */}
                <div className="grid grid-cols-2 gap-4 text-xs font-semibold">
                  <div>
                    <span className="text-[10px] text-muted-foreground font-medium uppercase">Age / Sex</span>
                    <p className="text-foreground mt-0.5">{selectedPatient.age} y / {selectedPatient.sex}</p>
                  </div>
                  <div>
                    <span className="text-[10px] text-muted-foreground font-medium uppercase">Status</span>
                    <p className="text-foreground mt-0.5 capitalize">{selectedPatient.is_archived === 1 ? "Archived" : "Active Case"}</p>
                  </div>
                </div>

                {/* Notes */}
                <div className="space-y-1">
                  <span className="text-[10px] text-muted-foreground font-medium uppercase">Clinical Intake Notes</span>
                  <p className="text-xs text-muted-foreground leading-relaxed bg-muted/20 border border-border/40 p-3 rounded-lg">
                    {selectedPatient.notes || "No clinician observations recorded for this patient profile."}
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Studies List */}
            <div className="space-y-4">
              <h4 className="text-sm font-bold uppercase tracking-wider text-foreground">Associated Studies History</h4>
              <Separator />

              {loadingStudies ? (
                <div className="p-8 text-center text-xs text-muted-foreground animate-pulse">
                  Querying studies...
                </div>
              ) : studies.length === 0 ? (
                <div className="p-8 text-center text-xs text-muted-foreground bg-muted/10 rounded-xl border border-dashed border-border">
                  No Chest Radiograph studies logged for this patient.
                </div>
              ) : (
                <div className="space-y-3">
                  {studies.map(study => {
                    const isTB = study.prediction?.toLowerCase().includes("tuberculosis") || study.is_tb;
                    return (
                      <Card key={study.study_id} className="border border-border bg-card rounded-xl shadow-none">
                        <CardContent className="p-4 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 text-xs font-medium">
                          <div className="space-y-1">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="font-mono font-bold text-foreground text-xs">{study.study_id}</span>
                              <Badge variant="outline" className={`${isTB ? "badge-tb border-amber-500/20 text-amber-600 bg-amber-500/5 font-bold" : "badge-normal"} rounded-full text-[9px] uppercase`}>
                                {study.prediction} ({(study.confidence * 100).toFixed(0)}%)
                              </Badge>
                            </div>
                            <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                              <span className="flex items-center gap-1">
                                <Calendar className="w-3.5 h-3.5" />
                                {new Date(study.timestamp).toLocaleDateString()}
                              </span>
                              <span>·</span>
                              <span>Modality: {study.metadata?.modality || "CR"}</span>
                            </div>
                          </div>

                          {onSelectStudy && (
                            <Button
                              onClick={() => onSelectStudy(study)}
                              size="sm"
                              className="rounded-full text-xs font-semibold h-8 gap-1 cursor-pointer"
                            >
                              Open Workbench <ExternalLink className="w-3.5 h-3.5" />
                            </Button>
                          )}
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="p-16 text-center text-xs text-muted-foreground bg-muted/10 border border-dashed border-border rounded-xl space-y-3">
            <Users className="w-8 h-8 text-muted-foreground/50 mx-auto" strokeWidth={1} />
            <h4 className="font-bold text-foreground">Select Patient Record</h4>
            <p className="max-w-xs mx-auto leading-relaxed">
              Select a clinical profile from the directory on the left to view demographics, edit records, and load associated X-ray studies.
            </p>
          </div>
        )}
      </div>

      {/* CREATE PATIENT MODAL */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-md border border-border bg-card rounded-xl overflow-hidden shadow-lg animate-scalein">
            <CardContent className="p-6 space-y-4">
              <div className="flex justify-between items-center">
                <h4 className="font-bold text-sm uppercase tracking-wider text-foreground">Create Patient Record</h4>
                <button onClick={() => setShowCreateModal(false)} className="text-muted-foreground hover:text-foreground cursor-pointer">
                  <X className="w-4 h-4" />
                </button>
              </div>
              <Separator />

              {formError && (
                <div className="p-3 bg-destructive/5 border border-destructive/20 rounded-xl text-xs text-destructive flex items-start gap-2">
                  <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  <span>{formError}</span>
                </div>
              )}

              <form onSubmit={handleCreatePatient} className="space-y-4 text-xs">
                {/* ID */}
                <div className="space-y-1.5">
                  <label className="text-muted-foreground font-semibold uppercase text-[10px]">Patient Identifier (MRN)</label>
                  <input
                    type="text"
                    value={formId}
                    onChange={e => setFormId(e.target.value)}
                    placeholder="e.g. PX-92812"
                    className="w-full px-4 py-2 border border-border rounded-full bg-card text-foreground focus:ring-1 focus:ring-primary outline-none h-10"
                    required
                  />
                </div>

                {/* Name */}
                <div className="space-y-1.5">
                  <label className="text-muted-foreground font-semibold uppercase text-[10px]">Patient Name</label>
                  <input
                    type="text"
                    value={formName}
                    onChange={e => setFormName(e.target.value)}
                    placeholder="e.g. Arjun Dev"
                    className="w-full px-4 py-2 border border-border rounded-full bg-card text-foreground focus:ring-1 focus:ring-primary outline-none h-10"
                    required
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {/* Age */}
                  <div className="space-y-1.5">
                    <label className="text-muted-foreground font-semibold uppercase text-[10px]">Age</label>
                    <input
                      type="text"
                      value={formAge}
                      onChange={e => setFormAge(e.target.value)}
                      placeholder="e.g. 45"
                      className="w-full px-4 py-2 border border-border rounded-full bg-card text-foreground focus:ring-1 focus:ring-primary outline-none h-10"
                    />
                  </div>

                  {/* Sex */}
                  <div className="space-y-1.5">
                    <label className="text-muted-foreground font-semibold uppercase text-[10px]">Sex</label>
                    <select
                      value={formSex}
                      onChange={e => setFormSex(e.target.value)}
                      className="w-full px-4 py-2 border border-border rounded-full bg-card text-foreground focus:ring-1 focus:ring-primary outline-none h-10 cursor-pointer appearance-none"
                    >
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                </div>

                {/* Notes */}
                <div className="space-y-1.5">
                  <label className="text-muted-foreground font-semibold uppercase text-[10px]">Clinical Notes</label>
                  <textarea
                    value={formNotes}
                    onChange={e => setFormNotes(e.target.value)}
                    placeholder="Enter patient notes, clinical records..."
                    rows={3}
                    className="w-full p-4 border border-border rounded-xl bg-card text-foreground focus:ring-1 focus:ring-primary outline-none"
                  />
                </div>

                <div className="flex justify-end gap-2 pt-2">
                  <Button type="button" variant="outline" onClick={() => setShowCreateModal(false)} className="rounded-full text-xs font-semibold h-10 px-5 cursor-pointer">
                    Cancel
                  </Button>
                  <Button type="submit" className="rounded-full text-xs font-semibold h-10 px-5 cursor-pointer">
                    Save Record
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>
        </div>
      )}

      {/* EDIT PATIENT MODAL */}
      {showEditModal && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-md border border-border bg-card rounded-xl overflow-hidden shadow-lg animate-scalein">
            <CardContent className="p-6 space-y-4">
              <div className="flex justify-between items-center">
                <h4 className="font-bold text-sm uppercase tracking-wider text-foreground">Edit Patient Record</h4>
                <button onClick={() => setShowEditModal(false)} className="text-muted-foreground hover:text-foreground cursor-pointer">
                  <X className="w-4 h-4" />
                </button>
              </div>
              <Separator />

              {formError && (
                <div className="p-3 bg-destructive/5 border border-destructive/20 rounded-xl text-xs text-destructive flex items-start gap-2">
                  <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  <span>{formError}</span>
                </div>
              )}

              <form onSubmit={handleUpdatePatient} className="space-y-4 text-xs">
                {/* ID (Disabled) */}
                <div className="space-y-1.5">
                  <label className="text-muted-foreground font-semibold uppercase text-[10px]">Patient Identifier (Disabled)</label>
                  <input
                    type="text"
                    value={formId}
                    disabled
                    className="w-full px-4 py-2 border border-border rounded-full bg-muted text-muted-foreground outline-none h-10 cursor-not-allowed"
                  />
                </div>

                {/* Name */}
                <div className="space-y-1.5">
                  <label className="text-muted-foreground font-semibold uppercase text-[10px]">Patient Name</label>
                  <input
                    type="text"
                    value={formName}
                    onChange={e => setFormName(e.target.value)}
                    placeholder="e.g. Arjun Dev"
                    className="w-full px-4 py-2 border border-border rounded-full bg-card text-foreground focus:ring-1 focus:ring-primary outline-none h-10"
                    required
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {/* Age */}
                  <div className="space-y-1.5">
                    <label className="text-muted-foreground font-semibold uppercase text-[10px]">Age</label>
                    <input
                      type="text"
                      value={formAge}
                      onChange={e => setFormAge(e.target.value)}
                      placeholder="e.g. 45"
                      className="w-full px-4 py-2 border border-border rounded-full bg-card text-foreground focus:ring-1 focus:ring-primary outline-none h-10"
                    />
                  </div>

                  {/* Sex */}
                  <div className="space-y-1.5">
                    <label className="text-muted-foreground font-semibold uppercase text-[10px]">Sex</label>
                    <select
                      value={formSex}
                      onChange={e => setFormSex(e.target.value)}
                      className="w-full px-4 py-2 border border-border rounded-full bg-card text-foreground focus:ring-1 focus:ring-primary outline-none h-10 cursor-pointer appearance-none"
                    >
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                </div>

                {/* Notes */}
                <div className="space-y-1.5">
                  <label className="text-muted-foreground font-semibold uppercase text-[10px]">Clinical Notes</label>
                  <textarea
                    value={formNotes}
                    onChange={e => setFormNotes(e.target.value)}
                    placeholder="Enter patient notes, clinical records..."
                    rows={3}
                    className="w-full p-4 border border-border rounded-xl bg-card text-foreground focus:ring-1 focus:ring-primary outline-none"
                  />
                </div>

                <div className="flex justify-end gap-2 pt-2">
                  <Button type="button" variant="outline" onClick={() => setShowEditModal(false)} className="rounded-full text-xs font-semibold h-10 px-5 cursor-pointer">
                    Cancel
                  </Button>
                  <Button type="submit" className="rounded-full text-xs font-semibold h-10 px-5 cursor-pointer">
                    Save Updates
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
