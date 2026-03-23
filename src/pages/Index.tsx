import ThreatAnalyzer from "@/components/ThreatAnalyzer";
import type { ScanResult } from "@/components/ThreatResult";

interface IndexProps {
  scans: ScanResult[];
  onScanComplete: (result: ScanResult) => void;
}

const Index = ({ scans, onScanComplete }: IndexProps) => {
  return (
    <div className="min-h-[calc(100vh-10rem)] flex items-center justify-center">
      <div className="w-full max-w-5xl">
        <ThreatAnalyzer onScanComplete={onScanComplete} />
      </div>
    </div>);

};

export default Index;