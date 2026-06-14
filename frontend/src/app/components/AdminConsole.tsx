import React, { useState, useEffect } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Shield, Key, Users, Activity, Database } from 'lucide-react';

export default function AdminConsole() {
  const [activeTab, setActiveTab] = useState<'users' | 'apikeys' | 'system'>('users');

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6 animate-fadein">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Enterprise Administration</h2>
          <p className="text-sm text-muted-foreground mt-1">Manage users, access control, and system health for ApolloDx.</p>
        </div>
        <Badge variant="outline" className="h-8 uppercase font-bold text-xs">Admin Access</Badge>
      </div>

      <div className="flex border-b border-border gap-6">
        {[
          { id: 'users', label: 'Users & Roles', icon: Users },
          { id: 'apikeys', label: 'API Keys', icon: Key },
          { id: 'system', label: 'System Health', icon: Activity },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`pb-3 text-sm font-semibold flex items-center gap-2 border-b-2 transition-colors ${
              activeTab === tab.id ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      <div className="pt-4">
        {activeTab === 'users' && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-bold">User Directory</h3>
              <button className="bg-primary text-primary-foreground px-4 py-2 rounded-lg text-xs font-bold">
                + Add User
              </button>
            </div>
            <Card className="shadow-none border border-border">
              <CardContent className="p-0">
                <table className="w-full text-sm text-left">
                  <thead className="bg-muted/50 text-xs uppercase text-muted-foreground font-bold">
                    <tr>
                      <th className="px-6 py-3">Username</th>
                      <th className="px-6 py-3">Role</th>
                      <th className="px-6 py-3">Institution</th>
                      <th className="px-6 py-3">Status</th>
                      <th className="px-6 py-3 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-border">
                      <td className="px-6 py-4 font-semibold">admin</td>
                      <td className="px-6 py-4"><Badge className="bg-purple-500/10 text-purple-500 hover:bg-purple-500/20">Admin</Badge></td>
                      <td className="px-6 py-4 text-muted-foreground">Central Hospital</td>
                      <td className="px-6 py-4"><span className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-emerald-500"></span> Active</span></td>
                      <td className="px-6 py-4 text-right"><button className="text-primary text-xs font-bold">Edit</button></td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="px-6 py-4 font-semibold">reviewer</td>
                      <td className="px-6 py-4"><Badge variant="outline">Radiologist</Badge></td>
                      <td className="px-6 py-4 text-muted-foreground">Central Hospital</td>
                      <td className="px-6 py-4"><span className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-emerald-500"></span> Active</span></td>
                      <td className="px-6 py-4 text-right"><button className="text-primary text-xs font-bold">Edit</button></td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 font-semibold">technician</td>
                      <td className="px-6 py-4"><Badge variant="secondary">Technician</Badge></td>
                      <td className="px-6 py-4 text-muted-foreground">Central Hospital</td>
                      <td className="px-6 py-4"><span className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-emerald-500"></span> Active</span></td>
                      <td className="px-6 py-4 text-right"><button className="text-primary text-xs font-bold">Edit</button></td>
                    </tr>
                  </tbody>
                </table>
              </CardContent>
            </Card>
          </div>
        )}

        {activeTab === 'apikeys' && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-bold">Developer API Keys</h3>
              <button className="bg-primary text-primary-foreground px-4 py-2 rounded-lg text-xs font-bold">
                Generate Key
              </button>
            </div>
            <Card className="shadow-none border border-border">
              <CardContent className="p-6 text-center space-y-3">
                <Key className="w-8 h-8 mx-auto text-muted-foreground" />
                <p className="text-sm font-semibold">No API Keys Generated</p>
                <p className="text-xs text-muted-foreground">Create a key to authenticate third-party integrations with the prediction API.</p>
              </CardContent>
            </Card>
          </div>
        )}

        {activeTab === 'system' && (
          <div className="grid grid-cols-2 gap-4">
            <Card className="shadow-none border border-border bg-card">
              <CardContent className="p-6 space-y-4">
                <div className="flex items-center gap-3">
                  <Database className="w-5 h-5 text-primary" />
                  <h4 className="font-bold text-sm">Storage Health</h4>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs font-semibold">
                    <span>Database SQLite</span>
                    <span className="text-emerald-500">Online</span>
                  </div>
                  <div className="flex justify-between text-xs font-semibold">
                    <span>Redis Queue</span>
                    <span className="text-amber-500">Warning (Local)</span>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card className="shadow-none border border-border bg-card">
              <CardContent className="p-6 space-y-4">
                <div className="flex items-center gap-3">
                  <Shield className="w-5 h-5 text-primary" />
                  <h4 className="font-bold text-sm">Security Audit</h4>
                </div>
                <div className="space-y-2 text-xs">
                  <p>• 15 failed logins in last 24h</p>
                  <p>• 3 new IP addresses detected</p>
                  <p>• All endpoints secured with RBAC</p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
