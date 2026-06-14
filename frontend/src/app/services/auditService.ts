export interface AuditLogEntry {
  timestamp: string;
  action: string;
}

export const auditService = {
  createDefaultLogs(filename: string): AuditLogEntry[] {
    const now = new Date();
    const subSeconds = (secs: number) => {
      const d = new Date(now.getTime() - secs * 1000);
      return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
    };

    return [
      { timestamp: subSeconds(90), action: `Image file [${filename}] loaded successfully` },
      { timestamp: subSeconds(75), action: "Inference engine pipeline initialized" },
      { timestamp: subSeconds(60), action: "Grad-CAM++ forward-backward pass complete" },
      { timestamp: subSeconds(45), action: "AI clinical observations mapped to coordinates" },
      { timestamp: subSeconds(30), action: "PACS record study tracking active" }
    ];
  }
};
