import { motion } from "framer-motion";
import { ShieldCheck, ShieldAlert, AlertTriangle } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export interface ScanResult {
  prediction: "threatening" | "non_threatening";
  confidence: number;
  method: string;
  indicators: string[];
  inputType: string;
  timestamp: Date;
  inputPreview: string;
}

const indicatorLabels: Record<string, string> = {
  intent: "Direct Intent",
  hostage_threat: "Hostage/Kidnapping",
  extortion_threat: "Extortion/Blackmail",
  doxing_threat: "Doxing",
  data_breach_threat: "Data Breach",
  cyber_threat: "Cyber Attack",
  implied_threat: "Veiled Threat",
  violence_keyword: "Violence Keyword",
  personal_pronoun: "Personal Pronoun",
  capability: "Stated Capability",
  specific_action: "Specific Action",
  temporal: "Time Reference",
  target: "Specific Target",
  weapon: "Weapon Reference",
  extremism: "Extremism Indicator",
  mass_target: "Mass Casualty Intent",
  vehicular_attack: "Vehicular Attack",
  finality: "Finality Language",
  desperation: "Desperation Signal",
  escalation: "Escalation Language",
  event_targeting: "Event Targeting",
  symbolic_targeting: "Symbolic Targeting",
  disruption_intent: "Disruption Intent",
  seriousness: "Seriousness Declaration",
  fixation: "Fixation Indicator",
  grievance: "Grievance Expression",
};

interface ThreatResultProps {
  result: ScanResult;
}

const ThreatResult = ({ result }: ThreatResultProps) => {
  const isThreat = result.prediction === "threatening";
  const confidencePct = result.confidence;

  return (
    <div className={`border-t-2 p-4 ${isThreat ? "border-t-destructive bg-destructive/5" : "border-t-primary bg-primary/5"}`}>
      <div className="flex items-start gap-3">
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", damping: 12 }}
        >
          {isThreat ? (
            <ShieldAlert className="w-8 h-8 text-destructive threat-pulse" />
          ) : (
            <ShieldCheck className="w-8 h-8 text-primary" />
          )}
        </motion.div>

        <div className="flex-1 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-bold font-mono text-sm">
              {isThreat ? "⚠ THREAT DETECTED" : "✓ NO THREAT DETECTED"}
            </h3>
            <span className={`text-xs font-mono font-bold ${isThreat ? "text-destructive" : "text-primary"}`}>
              {confidencePct.toFixed(1)}% confidence
            </span>
          </div>

          {/* Confidence bar */}
          <div className="h-1.5 rounded-full bg-muted overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${confidencePct}%` }}
              transition={{ duration: 0.8, ease: "easeOut" }}
              className={`h-full rounded-full ${isThreat ? "bg-destructive" : "bg-primary"}`}
            />
          </div>

          <div className="flex flex-wrap gap-2">
            {result.indicators.map((ind) => (
              <Badge
                key={ind}
                variant={isThreat ? "destructive" : "secondary"}
                className="text-xs font-mono"
              >
                {indicatorLabels[ind] || ind}
              </Badge>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ThreatResult;
