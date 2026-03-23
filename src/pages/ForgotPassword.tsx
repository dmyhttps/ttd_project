import { useState } from "react";
import { motion } from "framer-motion";
import { Shield, Mail, Loader2, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { supabase } from "@/integrations/supabase/client";
import { Link } from "react-router-dom";
import { toast } from "sonner";

const ForgotPassword = () => {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(false);

  const handleReset = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;

    setLoading(true);
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/reset-password`
    });
    setLoading(false);

    if (error) {
      toast.error(error.message);
    } else {
      setSent(true);
      toast.success("Check your email for a reset link");
    }
  };

  return (
    <div className="min-h-screen bg-background grid-bg flex items-center justify-center p-4 relative">
      <div className="fixed inset-0 scanline-overlay z-0" />
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative z-10 w-full max-w-md">
        
        <div className="rounded-lg border border-border bg-card overflow-hidden">
          <div className="border-b border-border p-6 text-center">
            <div className="flex items-center justify-center gap-2 mb-2">
              <Shield className="w-7 h-7 text-primary" />
              <h1 className="font-bold tracking-tight text-2xl">
                SENTINEL<span className="text-primary">​</span>
              </h1>
            </div>
            <p className="font-mono text-muted-foreground tracking-widest uppercase text-sm">
              Password Recovery
            </p>
          </div>

          {sent ?
          <div className="p-6 text-center space-y-3">
              <p className="text-sm text-foreground font-mono">Reset link sent to your email.</p>
              <Link to="/login">
                <Button variant="outline" size="sm" className="font-mono text-xs gap-1">
                  <ArrowLeft className="w-3 h-3" /> Back to login
                </Button>
              </Link>
            </div> :

          <form onSubmit={handleReset} className="p-6 space-y-4">
              <div className="space-y-2">
                <Label htmlFor="email" className="text-xs font-mono text-muted-foreground">EMAIL</Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <Input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="agent@sentinel"
                  className="pl-9 bg-muted border-border font-mono text-sm"
                  required
                  maxLength={255} />
                
                </div>
              </div>
              <Button type="submit" disabled={loading} className="w-full font-mono text-xs gap-2">
                {loading ?
              <><Loader2 className="w-3.5 h-3.5 animate-spin" /> SENDING</> :

              "SEND RESET LINK"
              }
              </Button>
              <div className="text-center">
                <Link to="/login" className="text-xs font-mono text-primary hover:underline">
                  Back to login
                </Link>
              </div>
            </form>
          }
        </div>
      </motion.div>
    </div>);

};

export default ForgotPassword;