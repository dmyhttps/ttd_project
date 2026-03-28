import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Shield, Lock, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { supabase } from "@/integrations/supabase/client";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";

const ResetPassword = () => {
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

 useEffect(() => {
  const hash = window.location.hash;
  const params = new URLSearchParams(window.location.search);
  const hasRecovery = hash.includes("type=recovery") || params.get("type") === "recovery";
  
  if (!hasRecovery) {
    toast.error("Invalid reset link");
    navigate("/login");
  }
}, [navigate]);
  const handleReset = async (e: React.FormEvent) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      toast.error("Passwords do not match");
      return;
    }
    if (password.length < 6) {
      toast.error("Password must be at least 6 characters");
      return;
    }

    setLoading(true);
    const { error } = await supabase.auth.updateUser({ password });
    setLoading(false);

    if (error) {
      toast.error(error.message);
    } else {
      toast.success("Password updated successfully");
      navigate("/");
    }
  };

  return (
    <div className="min-h-screen bg-background grid-bg flex items-center justify-center p-4 relative">
      <div className="fixed inset-0 scanline-overlay z-0" />
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative z-10 w-full max-w-md"
      >
        <div className="rounded-lg border border-border bg-card overflow-hidden">
          <div className="border-b border-border p-6 text-center">
            <Shield className="w-7 h-7 text-primary mx-auto mb-2" />
            <p className="text-xs font-mono text-muted-foreground tracking-widest uppercase">
              Set New Password
            </p>
          </div>

          <form onSubmit={handleReset} className="p-6 space-y-4">
            <div className="space-y-2">
              <Label className="text-xs font-mono text-muted-foreground">NEW PASSWORD</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="pl-9 bg-muted border-border font-mono text-sm"
                  required
                  minLength={6}
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label className="text-xs font-mono text-muted-foreground">CONFIRM PASSWORD</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="pl-9 bg-muted border-border font-mono text-sm"
                  required
                  minLength={6}
                />
              </div>
            </div>
            <Button type="submit" disabled={loading} className="w-full font-mono text-xs gap-2">
              {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : "UPDATE PASSWORD"}
            </Button>
          </form>
        </div>
      </motion.div>
    </div>
  );
};

export default ResetPassword;
