import { Shield } from "lucide-react";
import { motion } from "framer-motion";

const ThreatHeader = () => {
  return (
    <header className="border-b border-border px-6 py-4">
      <div className="flex items-center justify-between">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="gap-3 flex items-center justify-start"
        >
          <div className="relative">
            <Shield className="text-primary w-[50px] h-[50px]" />
            <div className="absolute inset-0 blur-md bg-primary/20 rounded-full border-0" />
          </div>
          <div>
            <h1 className="font-bold tracking-tight text-4xl">
              SENTINEL<span className="text-primary">​</span>
            </h1>
            <p className="text-muted-foreground font-mono tracking-widest uppercase text-base">
              Terrorism Threat Detection System
            </p>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="flex items-center gap-2"
        >
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-primary" />
          </span>
          <span className="font-mono text-muted-foreground hidden sm:inline text-base">SYSTEM ONLINE</span>
        </motion.div>
      </div>
    </header>
  );
};

export default ThreatHeader;
