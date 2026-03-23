import { useState } from "react";
import { motion } from "framer-motion";
import { Shield, Mail, Lock, Loader2, Eye, EyeOff, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { supabase } from "@/integrations/supabase/client";
import { Link } from "react-router-dom";
import { toast } from "sonner";

const Signup = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !password.trim()) return;

    if (password !== confirmPassword) {
      toast.error("Passwords do not match");
      return;
    }

    if (password.length < 6) {
      toast.error("Password must be at least 6 characters");
      return;
    }

    setLoading(true);
    const { error } = await supabase.auth.signUp({
      email,
      password,
      options: { emailRedirectTo: window.location.origin }
    });
    setLoading(false);

    if (error) {
      toast.error(error.message);
    } else {
      toast.success("Check your email to verify your account");
    }
  };

  return (
    <div className="min-h-screen bg-background grid-bg p-4 relative items-center justify-center flex flex-row">
      <div className="fixed inset-0 scanline-overlay z-0" />
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative z-10 w-full max-w-md">
        
        <div className="rounded-lg border border-border bg-card overflow-hidden">
          <div className="border-b border-border p-6 text-center">
            <div className="flex items-center justify-center gap-2 mb-2">
              <div className="relative">
                <Shield className="text-primary h-[40px] w-[40px]" />
                <div className="absolute inset-0 blur-md bg-primary/20 rounded-full" />
              </div>
              <h1 className="font-bold tracking-tight text-3xl">
                SENTINEL<span className="text-primary">​</span>
              </h1>
            </div>
            <p className="font-mono text-muted-foreground tracking-widest uppercase text-sm">
              Create Your Account
            </p>
          </div>

          <form onSubmit={handleSignup} className="p-6 space-y-4">
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

            <div className="space-y-2">
              <Label htmlFor="password" className="text-xs font-mono text-muted-foreground">PASSWORD</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="pl-9 pr-9 bg-muted border-border font-mono text-sm"
                  required
                  minLength={6} />
                
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground">
                  
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="confirmPassword" className="text-xs font-mono text-muted-foreground">CONFIRM PASSWORD</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  id="confirmPassword"
                  type={showPassword ? "text" : "password"}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="••••••••"
                  className="pl-9 bg-muted border-border font-mono text-sm"
                  required
                  minLength={6} />
                
              </div>
            </div>

            <Button type="submit" disabled={loading} className="w-full font-mono text-xs gap-2">
              {loading ?
              <><Loader2 className="w-3.5 h-3.5 animate-spin" /> CREATING ACCOUNT</> :

              <><User className="w-3.5 h-3.5" /> CREATE ACCOUNT</>
              }
            </Button>
          </form>

          <div className="border-t border-border px-6 py-4 text-center">
            <p className="text-muted-foreground font-mono text-sm">
              Already have access?{" "}
              <Link to="/login" className="text-primary hover:underline text-center text-base">
                Sign in
              </Link>
            </p>
          </div>
        </div>
      </motion.div>
    </div>);

};

export default Signup;