"use client";

import { useState, useEffect, useRef } from "react";
import { Bell, Check, Info, AlertTriangle, Eye } from "lucide-react";
import { Separator } from "@/components/ui/separator";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://projectmantra-nirikshon-backend.hf.space";

export interface NotificationItem {
  id: number;
  study_id: string;
  message: string;
  is_read: number;
  created_at: string;
}

export function NotificationsPanel() {
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const fetchNotifications = async () => {
    try {
      const response = await fetch(`${API_BASE}/notifications`, {
        credentials: "include"
      });
      if (response.ok) {
        const data = await response.json();
        setNotifications(data.notifications || []);
      }
    } catch (err) {
      console.error("Failed to load notifications:", err);
    }
  };

  const markAsRead = async (id: number) => {
    try {
      const response = await fetch(`${API_BASE}/notifications/${id}/read`, {
        method: "POST",
        credentials: "include"
      });
      if (response.ok) {
        setNotifications(prev => 
          prev.map(n => n.id === id ? { ...n, is_read: 1 } : n)
        );
      }
    } catch (err) {
      console.error("Failed to mark notification as read:", err);
    }
  };

  const markAllAsRead = async () => {
    const unread = notifications.filter(n => n.is_read === 0);
    for (const item of unread) {
      await markAsRead(item.id);
    }
  };

  useEffect(() => {
    fetchNotifications();
    // Poll notifications every 15 seconds
    const interval = setInterval(fetchNotifications, 15000);

    // Event listener to close dropdown on click outside
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);

    return () => {
      clearInterval(interval);
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const unreadCount = notifications.filter(n => n.is_read === 0).length;

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Bell Trigger Button */}
      <button
        onClick={() => setOpen(!open)}
        className="w-9 h-9 rounded-full bg-muted/40 hover:bg-muted/80 text-foreground flex items-center justify-center border border-border hover:border-primary/50 cursor-pointer relative transition-colors"
        title="Notifications Center"
      >
        <Bell className="w-4 h-4" strokeWidth={1.5} />
        {unreadCount > 0 && (
          <span className="absolute top-1.5 right-1.5 w-2.5 h-2.5 bg-destructive rounded-full border border-background animate-pulse" />
        )}
      </button>

      {/* Popover Dropdown list */}
      {open && (
        <div className="absolute right-0 mt-2.5 w-80 bg-card border border-border rounded-xl shadow-lg z-50 overflow-hidden animate-fadein">
          {/* Header */}
          <div className="p-4 flex items-center justify-between bg-muted/10">
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-bold text-foreground font-sans">System Alerts Hub</span>
              {unreadCount > 0 && (
                <span className="text-[10px] font-bold text-white bg-destructive px-1.5 py-0.5 rounded-full">
                  {unreadCount} new
                </span>
              )}
            </div>
            {unreadCount > 0 && (
              <button
                onClick={markAllAsRead}
                className="text-[10px] font-semibold text-primary hover:underline cursor-pointer"
              >
                Mark all read
              </button>
            )}
          </div>
          <Separator />

          {/* List Content */}
          <div className="max-h-[300px] overflow-y-auto divide-y divide-border scrollbar-none">
            {notifications.length === 0 ? (
              <div className="p-6 text-center text-xs text-muted-foreground flex flex-col items-center gap-2">
                <Info className="w-6 h-6 text-muted-foreground/60" strokeWidth={1} />
                <p>No active notification alerts</p>
              </div>
            ) : (
              notifications.map(item => {
                const isTBAlert = item.message.toLowerCase().includes("tuberculosis") || item.message.toLowerCase().includes("tb");
                return (
                  <div 
                    key={item.id} 
                    className={`p-3.5 flex items-start gap-3 transition-colors hover:bg-muted/10 ${
                      item.is_read === 0 ? "bg-primary/5 border-l-2 border-primary" : ""
                    }`}
                  >
                    <div className="mt-0.5">
                      {isTBAlert ? (
                        <AlertTriangle className="w-4 h-4 text-amber-500" strokeWidth={1.5} />
                      ) : (
                        <Info className="w-4 h-4 text-primary" strokeWidth={1.5} />
                      )}
                    </div>
                    
                    <div className="flex-1 space-y-1">
                      <p className="text-[11px] font-medium text-foreground leading-relaxed">
                        {item.message}
                      </p>
                      <span className="text-[9px] text-muted-foreground font-mono block">
                        {new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>

                    {item.is_read === 0 && (
                      <button
                        onClick={() => markAsRead(item.id)}
                        className="text-muted-foreground hover:text-primary transition-colors cursor-pointer w-5 h-5 rounded-full flex items-center justify-center hover:bg-muted/40"
                        title="Mark as Read"
                      >
                        <Check className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}
