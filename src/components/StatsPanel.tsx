import { Shield, AlertTriangle, FileSearch, Activity } from "lucide-react";
import { motion } from "framer-motion";

interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  trend?: string;
  delay: number;
}

const StatCard = ({ icon, label, value, trend, delay }: StatCardProps) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay }}
    className="rounded-lg border border-border bg-card p-4 relative overflow-hidden"
  >
    <div className="flex items-center justify-between mb-2">
      <span className="text-muted-foreground">{icon}</span>
      {trend && <span className="text-xs font-mono text-primary">{trend}</span>}
    </div>
    <p className="text-2xl font-bold font-mono">{value}</p>
    <p className="text-muted-foreground mt-1 text-base">{label}</p>
  </motion.div>
);

interface StatsPanelProps {
  totalScans: number;
  threatsDetected: number;
}

const StatsPanel = ({ totalScans, threatsDetected }: StatsPanelProps) => {
  return (
    <div className="grid grid-cols-3 gap-4">
      <StatCard
        icon={<FileSearch className="w-4 h-4" />}
        label="Total Scans"
        value={totalScans}
        delay={0.1}
      />
      <StatCard
        icon={<AlertTriangle className="w-4 h-4" />}
        label="Threats Found"
        value={threatsDetected}
        trend={threatsDetected > 0 ? "⚠" : undefined}
        delay={0.15}
      />
      <StatCard
        icon={<Activity className="w-4 h-4" />}
        label="Detection Methods"
        value="BERT + Rules"
        delay={0.2}
      />
    </div>
  );
};

export default StatsPanel;
