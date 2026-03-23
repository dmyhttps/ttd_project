import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { ShieldAlert, ShieldCheck, Clock, Search, Filter } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import type { ScanResult } from "@/components/ThreatResult";

interface HistoryProps {
  scans: ScanResult[];
}

const History = ({ scans }: HistoryProps) => {
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [resultFilter, setResultFilter] = useState<string>("all");
  const [dateFilter, setDateFilter] = useState<string>("all");

  const filtered = useMemo(() => {
    return scans.filter((scan) => {
      // Search
      if (search && !scan.inputPreview.toLowerCase().includes(search.toLowerCase())) {
        return false;
      }
      // Input type
      if (typeFilter !== "all" && scan.inputType !== typeFilter) {
        return false;
      }
      // Result
      if (resultFilter !== "all" && scan.prediction !== resultFilter) {
        return false;
      }
      // Date
      if (dateFilter !== "all") {
        const now = new Date();
        const scanDate = new Date(scan.timestamp);
        const diffMs = now.getTime() - scanDate.getTime();
        const diffHours = diffMs / (1000 * 60 * 60);
        if (dateFilter === "1h" && diffHours > 1) return false;
        if (dateFilter === "24h" && diffHours > 24) return false;
        if (dateFilter === "7d" && diffHours > 168) return false;
      }
      return true;
    });
  }, [scans, search, typeFilter, resultFilter, dateFilter]);

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h2 className="text-2xl font-bold font-mono tracking-tight">SCAN HISTORY</h2>
        <p className="text-sm text-muted-foreground font-mono mt-1">
          {scans.length} total scan(s)
        </p>
      </motion.div>

      {/* Filters */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="flex flex-col sm:flex-row gap-3"
      >
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search scans..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9 bg-muted border-border font-mono text-sm"
          />
        </div>
        <Select value={typeFilter} onValueChange={setTypeFilter}>
          <SelectTrigger className="w-full sm:w-[140px] bg-muted border-border font-mono text-xs">
            <SelectValue placeholder="Input Type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Types</SelectItem>
            <SelectItem value="text">Text</SelectItem>
            <SelectItem value="pdf">PDF</SelectItem>
            <SelectItem value="file">File</SelectItem>
            <SelectItem value="audio">Audio</SelectItem>
          </SelectContent>
        </Select>
        <Select value={resultFilter} onValueChange={setResultFilter}>
          <SelectTrigger className="w-full sm:w-[160px] bg-muted border-border font-mono text-xs">
            <SelectValue placeholder="Result" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Results</SelectItem>
            <SelectItem value="threatening">Threatening</SelectItem>
            <SelectItem value="non_threatening">Non-Threatening</SelectItem>
          </SelectContent>
        </Select>
        <Select value={dateFilter} onValueChange={setDateFilter}>
          <SelectTrigger className="w-full sm:w-[140px] bg-muted border-border font-mono text-xs">
            <SelectValue placeholder="Date" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Time</SelectItem>
            <SelectItem value="1h">Last Hour</SelectItem>
            <SelectItem value="24h">Last 24h</SelectItem>
            <SelectItem value="7d">Last 7 Days</SelectItem>
          </SelectContent>
        </Select>
      </motion.div>

      {/* Results */}
      {filtered.length === 0 ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="rounded-lg border border-border bg-card p-12 text-center"
        >
          <Filter className="w-8 h-8 text-muted-foreground mx-auto mb-3" />
          <p className="text-sm text-muted-foreground font-mono">
            {scans.length === 0 ? "No scans yet" : "No scans match your filters"}
          </p>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="rounded-lg border border-border bg-card overflow-hidden"
        >
          <div className="divide-y divide-border">
            {filtered.map((scan, i) => {
              const isThreat = scan.prediction === "threatening";
              return (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.03 }}
                  className="px-4 py-4 flex items-center gap-3 hover:bg-muted/50 transition-colors"
                >
                  {isThreat ? (
                    <ShieldAlert className="w-5 h-5 text-destructive flex-shrink-0" />
                  ) : (
                    <ShieldCheck className="w-5 h-5 text-primary flex-shrink-0" />
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-mono truncate text-foreground">
                      {scan.inputPreview || "—"}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      <Badge variant="outline" className="text-xs font-mono">
                        {scan.inputType}
                      </Badge>
                      {scan.indicators.length > 0 && (
                        <span className="text-xs text-muted-foreground">
                          {scan.indicators.join(", ")}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="text-right flex-shrink-0">
                    <p className={`text-sm font-mono font-bold ${isThreat ? "text-destructive" : "text-primary"}`}>
                      {scan.confidence.toFixed(0)}%
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {scan.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </p>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default History;
