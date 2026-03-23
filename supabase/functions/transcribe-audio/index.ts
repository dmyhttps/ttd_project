import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) throw new Error("LOVABLE_API_KEY is not configured");

    const formData = await req.formData();
    const audioFile = formData.get("audio") as File;
    if (!audioFile) throw new Error("No audio file provided");

    // Convert to base64
    const arrayBuffer = await audioFile.arrayBuffer();
    const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));

    // Determine MIME type
    const ext = audioFile.name.split(".").pop()?.toLowerCase() || "wav";
    const mimeMap: Record<string, string> = {
      mp3: "audio/mpeg",
      wav: "audio/wav",
      m4a: "audio/mp4",
      ogg: "audio/ogg",
      webm: "audio/webm",
      aac: "audio/aac",
    };
    const mimeType = mimeMap[ext] || "audio/wav";

    const response = await fetch(
      "https://ai.gateway.lovable.dev/v1/chat/completions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${LOVABLE_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "google/gemini-2.5-flash",
          messages: [
            {
              role: "system",
              content:
                "You are a precise audio transcription tool. Transcribe the audio exactly as spoken. Output ONLY the transcription text, nothing else. No commentary, no labels, no formatting. If you cannot understand the audio or it is silence, output an empty string.",
            },
            {
              role: "user",
              content: [
                {
                  type: "input_audio",
                  input_audio: {
                    data: base64,
                    format: ext === "mp3" ? "mp3" : "wav",
                  },
                },
                {
                  type: "text",
                  text: "Transcribe this audio exactly as spoken.",
                },
              ],
            },
          ],
        }),
      }
    );

    if (!response.ok) {
      if (response.status === 429) {
        return new Response(
          JSON.stringify({ error: "Rate limit exceeded, please try again later." }),
          { status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      if (response.status === 402) {
        return new Response(
          JSON.stringify({ error: "Payment required, please add credits." }),
          { status: 402, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      const errText = await response.text();
      console.error("AI gateway error:", response.status, errText);
      throw new Error(`AI gateway error: ${response.status}`);
    }

    const data = await response.json();
    const transcript =
      data.choices?.[0]?.message?.content?.trim() || "";

    return new Response(JSON.stringify({ transcript }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("transcribe-audio error:", e);
    return new Response(
      JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
