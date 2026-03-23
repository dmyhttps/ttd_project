import { motion } from "framer-motion";
import { LogOut, User, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";

const SettingsPage = () => {
  const { user, signOut } = useAuth();

  return (
    <div className="space-y-8 max-w-3xl">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h2 className="text-4xl font-bold font-mono tracking-tight">SETTINGS</h2>
        <p className="text-base text-muted-foreground font-mono mt-2">Manage your account</p>
      </motion.div>

      {/* Account section */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="rounded-lg border border-border bg-card overflow-hidden"
      >
        <div className="border-b border-border px-6 py-4 flex items-center gap-3">
          <User className="w-5 h-5 text-muted-foreground" />
          <span className="text-base font-mono font-medium">ACCOUNT</span>
        </div>
        <div className="p-6 space-y-5">
          <div>
            <label className="text-sm font-mono text-muted-foreground uppercase">Email</label>
            <p className="text-base font-mono mt-1">{user?.email || "—"}</p>
          </div>
          <div>
            <label className="text-sm font-mono text-muted-foreground uppercase">User ID</label>
            <p className="text-sm font-mono mt-1 text-muted-foreground truncate">{user?.id || "—"}</p>
          </div>
        </div>
      </motion.div>

      {/* System Info */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="rounded-lg border border-border bg-card overflow-hidden"
      >
        <div className="border-b border-border px-6 py-4 flex items-center gap-3">
          <Shield className="w-5 h-5 text-muted-foreground" />
          <span className="text-base font-mono font-medium">SYSTEM</span>
        </div>
        <div className="p-6 space-y-3">
          <div className="flex justify-between">
            <span className="text-sm font-mono text-muted-foreground">Version</span>
            <span className="text-sm font-mono">1.0.0</span>
          </div>
          <div className="flex justify-between">
            <span className="text-sm font-mono text-muted-foreground">Detection Engine</span>
            <span className="text-sm font-mono">BERT + Rules</span>
          </div>
        </div>
      </motion.div>

      {/* Logout */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Button
          onClick={signOut}
          variant="destructive"
          className="gap-2 font-mono text-sm w-full sm:w-auto h-12 px-6"
        >
          <LogOut className="w-5 h-5" />
          SIGN OUT
        </Button>
      </motion.div>
    </div>
  );
};

export default SettingsPage;
