import { motion } from "framer-motion";
import { Crosshair, Users, Lock, Globe, Eye, Swords } from "lucide-react";

const categories = [
{ icon: Swords, label: "Direct Violence", desc: "Kill, attack, bomb threats" },
{ icon: Users, label: "Hostage/Kidnapping", desc: "Abduction & ransom demands" },
{ icon: Lock, label: "Extortion", desc: "Blackmail & data ransom" },
{ icon: Eye, label: "Doxing", desc: "Personal info exposure" },
{ icon: Globe, label: "Cyber Threats", desc: "DDoS, ransomware, hacking" },
{ icon: Crosshair, label: "Veiled Threats", desc: "Implied & indirect threats" }];


const ThreatCategories = () =>
<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ delay: 0.5 }}
  className="rounded-lg border border-border bg-card overflow-hidden">
  
    <div className="border-b border-border px-4 py-3">
      <span className="text-sm font-mono font-medium">​</span>
    </div>
    















  
  </motion.div>;


export default ThreatCategories;