import { motion } from "framer-motion";
import { ShieldAlert, ShieldCheck, Clock } from "lucide-react";
import type { ScanResult } from "./ThreatResult";

interface ScanHistoryProps {
  scans: ScanResult[];
}

const ScanHistory = ({ scans }: ScanHistoryProps) => {
  if (scans.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="rounded-lg border border-border bg-card p-6 text-center"
      >
        <Clock className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
        <p className="text-sm text-muted-foreground font-mono">No scans yet</p>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 }}
      className="rounded-lg border border-border bg-card overflow-hidden"
    >
      <div className="border-b border-border px-4 py-3 flex items-center gap-2">
        <Clock className="w-4 h-4 text-muted-foreground" />
        <span className="text-sm font-mono font-medium">SCAN HISTORY</span>
        <span className="ml-auto text-xs text-muted-foreground font-mono">{scans.length} scan(s)</span>
      </div>
      <div className="divide-y divide-border max-h-[400px] overflow-y-auto">
        {scans.map((scan, i) => {
          const isThreat = scan.prediction === "threatening";
          return (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
              className="px-4 py-3 flex items-center gap-3 hover:bg-muted/50 transition-colors"
            >
              {isThreat ? (
                <ShieldAlert className="w-4 h-4 text-destructive flex-shrink-0" />
              ) : (
                <ShieldCheck className="w-4 h-4 text-primary flex-shrink-0" />
              )}
              <div className="flex-1 min-w-0">
                <p className="text-xs font-mono truncate text-foreground">
                  {scan.inputPreview || "—"}
                </p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {scan.indicators.length > 0 ? scan.indicators.join(", ") : "clean"}
                </p>
              </div>
              <div className="text-right flex-shrink-0">
                <p className={`text-xs font-mono font-bold ${isThreat ? "text-destructive" : "text-primary"}`}>
                  {scan.confidence.toFixed(0)}%
                </p>
                <p className="text-xs text-muted-foreground">
                  {scan.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>
              </div>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
};

export default ScanHistory;
