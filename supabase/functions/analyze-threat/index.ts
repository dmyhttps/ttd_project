import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) throw new Error("LOVABLE_API_KEY is not configured");

    const { text, ruleResult } = await req.json();
    if (!text || typeof text !== "string") {
      throw new Error("No text provided");
    }

    // Guardrail: reject raw PDF/binary payloads masquerading as text
    const looksLikeRawPdf =
      text.startsWith("%PDF-") ||
      (/\bobj\b/i.test(text) && /\bendobj\b/i.test(text) && /\bxref\b/i.test(text));

    if (looksLikeRawPdf) {
      return new Response(
        JSON.stringify({
          error: "Unreadable PDF binary detected. Extract text from the PDF before analysis.",
        }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }

    const systemPrompt = `You are a specialized threat detection AI for a security platform called Sentinel. Your job is to classify text as either "threatening" or "non_threatening".

IMPORTANT GUIDELINES:
- Flag text as "threatening" when there is genuine, credible intent of violence, harm, extortion, kidnapping, terrorism, or coordinated attack planning.
- Also flag as "threatening" for contextual/veiled threats when multiple risk cues combine (e.g., grievance escalation + upcoming event/place reference + ominous warning like "they'll understand soon" + stated shift from words to action).
- Do NOT flag: figurative language, idioms ("kill it", "bombing the test"), gaming references, fiction/stories, historical discussion, or clearly hypothetical scenarios.
- Consider context carefully. "I'm going to kill this presentation" is NOT a threat. "I'm going to kill everyone at the school tomorrow" IS a threat.
- Look for: specific targets, timelines, capability (weapons/vehicles/tools), first-person intent, fixation, coercion, and disruption intent.
- A negated threat ("I would never hurt anyone") is NOT a threat.

The rule-based system flagged this as ambiguous. Provide your analysis.

You MUST respond using the provided tool.`;

    const MAX_AI_CHARS = 12000;
    const textForAI =
      text.length > MAX_AI_CHARS
        ? `${text.slice(0, MAX_AI_CHARS / 2)}\n\n[...TRUNCATED ${text.length - MAX_AI_CHARS} CHARS...]\n\n${text.slice(-MAX_AI_CHARS / 2)}`
        : text;

    const models = ["google/gemini-2.5-flash", "google/gemini-2.5-flash-lite"];
    let lastError: Error | null = null;

    for (const model of models) {
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 25000);

        const response = await fetch(
          "https://ai.gateway.lovable.dev/v1/chat/completions",
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${LOVABLE_API_KEY}`,
              "Content-Type": "application/json",
            },
            signal: controller.signal,
            body: JSON.stringify({
              model,
              messages: [
                { role: "system", content: systemPrompt },
                {
                  role: "user",
                  content: `Analyze this text for threats. The rule-based system produced: ${JSON.stringify(ruleResult)}.\n\nText to analyze:\n"${textForAI}"`,
                },
              ],
              tools: [
                {
                  type: "function",
                  function: {
                    name: "classify_threat",
                    description:
                      "Classify the given text as threatening or non-threatening with detailed analysis.",
                    parameters: {
                      type: "object",
                      properties: {
                        prediction: {
                          type: "string",
                          enum: ["threatening", "non_threatening"],
                          description: "The threat classification",
                        },
                        confidence: {
                          type: "number",
                          description:
                            "Confidence percentage (0-100). Use 90+ only for very clear cases.",
                        },
                        indicators: {
                          type: "array",
                          items: { type: "string" },
                          description:
                            "List of threat indicators found (e.g. 'intent', 'weapon', 'target', 'temporal', 'extortion_threat', 'hostage_threat', 'doxing_threat') or empty if non-threatening",
                        },
                        reasoning: {
                          type: "string",
                          description:
                            "Brief explanation of why this was classified this way",
                        },
                      },
                      required: [
                        "prediction",
                        "confidence",
                        "indicators",
                        "reasoning",
                      ],
                      additionalProperties: false,
                    },
                  },
                },
              ],
              tool_choice: {
                type: "function",
                function: { name: "classify_threat" },
              },
            }),
          }
        );

        clearTimeout(timeout);

        if (!response.ok) {
          if (response.status === 429 || response.status === 402) {
            await response.text(); // consume body
            lastError = new Error(`AI gateway error: ${response.status}`);
            continue; // try next model
          }
          const errText = await response.text();
          console.error(`AI gateway error (${model}):`, response.status, errText);
          lastError = new Error(`AI gateway error: ${response.status}`);
          continue;
        }

        const data = await response.json();
        const toolCall = data.choices?.[0]?.message?.tool_calls?.[0];
        if (!toolCall) {
          lastError = new Error("No tool call in AI response");
          continue;
        }

        const result = JSON.parse(toolCall.function.arguments);

        return new Response(
          JSON.stringify({
            prediction: result.prediction,
            confidence: Math.min(99, Math.max(1, result.confidence)),
            method: "ai_analysis",
            indicators: result.indicators || [],
            reasoning: result.reasoning,
          }),
          {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          }
        );
      } catch (modelErr) {
        console.error(`Model ${model} failed:`, modelErr);
        lastError = modelErr instanceof Error ? modelErr : new Error(String(modelErr));
        continue;
      }
    }

    // All models failed
    throw lastError || new Error("All AI models failed");
  } catch (e) {
    console.error("analyze-threat error:", e);
    return new Response(
      JSON.stringify({
        error: e instanceof Error ? e.message : "Unknown error",
      }),
      {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      }
    );
  }
});
