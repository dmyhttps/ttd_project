import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, FileText, Mic, FileUp, Loader2, Shield, AudioLines, Brain, Cpu } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ThreatResult, { type ScanResult } from "./ThreatResult";
import { analyzeTextRules } from "@/lib/threatAnalysis";
import { extractTextFromFile } from "@/lib/fileTextExtraction";

interface ThreatAnalyzerProps {
  onScanComplete: (result: ScanResult) => void;
}

const ThreatAnalyzer = ({ onScanComplete }: ThreatAnalyzerProps) => {
  const [text, setText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [analysisMethod, setAnalysisMethod] = useState<string>("");
  const [activeTab, setActiveTab] = useState<string>("text");

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setIsAnalyzing(true);
    setResult(null);
    setAnalysisMethod("");

    // Determine the real input type based on active tab and file name
    const resolvedInputType =
      activeTab === "audio" ? "audio"
      : activeTab === "file" && fileName
        ? (fileName.toLowerCase().endsWith(".pdf") ? "pdf" : "file")
        : "text";

    try {
      // Step 1: Fast rule-based analysis
      const ruleAnalysis = analyzeTextRules(text, resolvedInputType);

      if (ruleAnalysis.confidence_level === 'clear_threat' || ruleAnalysis.confidence_level === 'clear_safe') {
        // High confidence — use rule result directly
        setAnalysisMethod(ruleAnalysis.confidence_level === 'clear_threat' ? 'Rule Engine (High Confidence)' : 'Rule Engine');
        setResult(ruleAnalysis.scanResult);
        onScanComplete(ruleAnalysis.scanResult);
        return;
      }

      // Step 2: Ambiguous — escalate to AI
      setAnalysisMethod("AI Analysis (Ambiguous Input)");
      const { data, error } = await supabase.functions.invoke("analyze-threat", {
        body: {
          text,
          ruleResult: {
            prediction: ruleAnalysis.scanResult.prediction,
            confidence: ruleAnalysis.scanResult.confidence,
            indicators: ruleAnalysis.scanResult.indicators,
          },
        },
      });

      if (error) throw error;
      if (data?.error) throw new Error(data.error);

      const aiResult: ScanResult = {
        prediction: data.prediction,
        confidence: data.confidence,
        method: "ai_analysis",
        indicators: data.indicators || [],
        inputType: resolvedInputType,
        timestamp: new Date(),
        inputPreview: text.slice(0, 120),
      };

      setResult(aiResult);
      onScanComplete(aiResult);
    } catch (err: any) {
      console.error("Analysis error:", err);
      // Fallback to rule-based result on AI failure
      const fallback = analyzeTextRules(text, resolvedInputType);
      setAnalysisMethod("Rule Engine (AI Fallback)");
      setResult(fallback.scanResult);
      onScanComplete(fallback.scanResult);
      toast.error("AI analysis unavailable, used rule-based fallback");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFileRead = async (file: File) => {
    setFileName(file.name);
    setText("");

    try {
      const extractedText = await extractTextFromFile(file);
      if (!extractedText.trim()) {
        toast.warning("No readable text found in file");
        return;
      }

      setText(extractedText);
      if (file.name.toLowerCase().endsWith('.pdf')) {
        toast.success("PDF text extracted successfully");
      }
    } catch (err: any) {
      console.error("File read error:", err);
      toast.error(err?.message || "Failed to read file content");
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="border-border bg-card overflow-hidden py-0 border-0 mx-0 my-0 px-0 rounded-xl">
      
      <div className="border-b border-border px-4 py-3 gap-2 flex items-center justify-start">
        <div className="w-2 h-2 rounded-full bg-primary" />
        <span className="font-mono font-medium text-lg">THREAT ANALYZER</span>
        <span className="ml-auto text-xs font-mono text-muted-foreground flex items-center gap-1">
          <Brain className="w-3 h-3" /> BERT Model
        </span>
      </div>

      <Tabs defaultValue="text" onValueChange={setActiveTab} className="p-4">
        <TabsList className="bg-muted border border-border mb-4">
          <TabsTrigger value="text" className="gap-2 text-sm font-mono data-[state=active]:bg-secondary">
            <Send className="w-4 h-4" /> Text
          </TabsTrigger>
          <TabsTrigger value="file" className="gap-2 text-sm font-mono data-[state=active]:bg-secondary">
            <FileText className="w-4 h-4" /> File
          </TabsTrigger>
          <TabsTrigger value="audio" className="gap-2 text-sm font-mono data-[state=active]:bg-secondary">
            <AudioLines className="w-4 h-4" /> Audio
          </TabsTrigger>
        </TabsList>

        <TabsContent value="text" className="space-y-4">
          <Textarea
            placeholder="Enter text to analyze for threats..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="min-h-[200px] bg-muted border-border font-mono text-sm resize-none placeholder:text-muted-foreground/50" />
        </TabsContent>

        <TabsContent value="file" className="space-y-4">
          <div className="border-2 border-dashed border-border rounded-lg p-8 text-center">
            <FileUp className="w-8 h-8 text-muted-foreground mx-auto mb-3" />
            <p className="text-sm text-muted-foreground mb-3">
              Upload .txt or .pdf files for analysis
            </p>
            <label className="cursor-pointer">
              <input
                type="file"
                accept=".txt,.pdf"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFileRead(file);
                }} />
              <Button variant="outline" size="sm" className="font-mono text-xs" asChild>
                <span>Choose File</span>
              </Button>
            </label>
            {fileName && <p className="text-xs text-primary mt-2 font-mono">📎 {fileName}</p>}
          </div>
          {text && (
            <div className="bg-muted rounded-md p-3 max-h-32 overflow-y-auto">
              <p className="text-xs font-mono text-muted-foreground">{text.slice(0, 500)}</p>
            </div>
          )}
        </TabsContent>

        <TabsContent value="audio" className="space-y-4">
          <div className="border-2 border-dashed border-border rounded-lg p-8 text-center">
            <Mic className="w-8 h-8 text-muted-foreground mx-auto mb-3" />
            <p className="text-sm text-muted-foreground mb-3">
              {isTranscribing ? "Transcribing audio..." : "Upload audio files for transcription & analysis"}
            </p>
            <label className="cursor-pointer">
              <input
                type="file"
                accept=".mp3,.wav,.m4a,.ogg,.webm,.aac"
                className="hidden"
                onChange={async (e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    setFileName(file.name);
                    setAudioFile(file);
                    setIsTranscribing(true);
                    setText("");
                    try {
                      const formData = new FormData();
                      formData.append("audio", file);
                      const { data, error } = await supabase.functions.invoke(
                        "transcribe-audio",
                        { body: formData }
                      );
                      if (error) throw error;
                      if (data?.error) throw new Error(data.error);
                      const transcript = data?.transcript || "";
                      if (transcript) {
                        setText(transcript);
                        toast.success("Audio transcribed successfully");
                      } else {
                        setText("");
                        toast.warning("Could not extract speech from audio");
                      }
                    } catch (err: any) {
                      console.error("Transcription error:", err);
                      toast.error(err?.message || "Failed to transcribe audio");
                    } finally {
                      setIsTranscribing(false);
                    }
                  }
                }} />
              <Button variant="outline" size="sm" className="font-mono text-xs" asChild>
                <span>Choose Audio File</span>
              </Button>
            </label>
            {fileName && fileName.match(/\.(mp3|wav|m4a|ogg|webm|aac)$/i) && (
              <p className="text-xs text-primary mt-2 font-mono">🎵 {fileName}</p>
            )}
          </div>
          {text && (
            <div className="bg-muted rounded-md p-3 max-h-32 overflow-y-auto">
              <p className="text-xs font-mono text-muted-foreground">{text.slice(0, 500)}</p>
            </div>
          )}
        </TabsContent>

        <div className="flex items-center justify-between mt-4">
          <p className="text-muted-foreground font-mono text-base text-left">
            {text.length > 0 ? `${text.length} chars` : ""}
          </p>
          <Button
            onClick={handleAnalyze}
            disabled={!text.trim() || isAnalyzing}
            size="lg"
            className="gap-2 font-mono text-base">
            {isAnalyzing ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                ANALYZING
              </>
            ) : (
              <>
                <Shield className="w-5 h-5" />
                ANALYZE THREAT
              </>
            )}
          </Button>
        </div>
      </Tabs>

      {/* Scanning animation */}
      <AnimatePresence>
        {isAnalyzing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="relative h-1 bg-muted overflow-hidden">
            <motion.div
              className="absolute inset-y-0 left-0 bg-primary/60"
              initial={{ width: "0%" }}
              animate={{ width: "100%" }}
              transition={{ duration: 1.5, ease: "easeInOut" }} />
          </motion.div>
        )}
      </AnimatePresence>


      {/* Result */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}>
            <ThreatResult result={result} />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default ThreatAnalyzer;
