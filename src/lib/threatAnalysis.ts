import type { ScanResult } from "@/components/ThreatResult";

// ── Idioms & figurative language that should NOT trigger threats ──
const SAFE_IDIOMS = [
  /\bkill(ing|ed|s)?\s+(it|time|the\s+game|the\s+mood|the\s+vibe|two\s+birds)/i,
  /\bdead(ly)?\s+(tired|serious|wrong|right|silence|end|line|pan|lock)/i,
  /\bbomb(ed|ing|s)?\s+(the\s+test|the\s+exam|the\s+interview|it)/i,
  /\bshoot(ing)?\s+(for|the\s+breeze|hoops|a\s+message|an?\s+email|photos?|pictures?|video)/i,
  /\bblew\s+(up|it|my\s+mind)/i,
  /\bblast(ing|ed)?\s+(music|song|off)/i,
  /\btake\s+a\s+stab\s+at/i,
  /\bstab(bing)?\s+in\s+the\s+(back|dark)/i,
  /\bdestroy(ed|ing|s)?\s+(the\s+competition|the\s+game|it|that|him|her|them)\b(?!.{0,20}(physically|literally))/i,
  /\bslaughter(ed|ing)?\s+(the\s+competition|it|them|that)/i,
  /\byou('re|r|\s+are)\s+killing\s+me/i,
  /\bto\s+die\s+for\b/i,
  /\bdrop\s+dead\s+gorgeous/i,
  /\bkiller\s+(app|feature|look|smile|deal|instinct)/i,
  /\bbullet\s+(point|proof|journal)/i,
  /\btrigger(ed|ing|s)?\s+(warning|a\s+build|a\s+deploy|an?\s+event|the\s+pipeline)/i,
  /\bexplod(e|ed|ing|es)?\s+(with\s+joy|with\s+laughter|in\s+popularity)/i,
  /\bmurder(ed|ing|s)?\s+(that\s+song|the\s+dance|on\s+the\s+field)/i,
  /\battack(ed|ing|s)?\s+(the\s+problem|this\s+issue|the\s+project|the\s+challenge)/i,
  /\bpoisoned?\s+(the\s+well|chalice|pill)/i,
  /\bharm(less|ony|onica)/i,
];

// ── Negation patterns ──
const NEGATION_WINDOW = /\b(not|never|wouldn't|won't|don't|doesn't|didn't|can't|cannot|no\s+intention|no\s+plan|no\s+desire|would\s+never|will\s+never|do\s+not|should\s+not|shouldn't|isn't|aren't|wasn't|weren't)\b/i;

function hasNearbyNegation(text: string, keywordIndex: number): boolean {
  const windowStart = Math.max(0, keywordIndex - 60);
  const windowEnd = Math.min(text.length, keywordIndex + 10);
  const window = text.slice(windowStart, windowEnd);
  return NEGATION_WINDOW.test(window);
}

// ── Enhanced keywords with severity weighting ──
const VIOLENCE_KEYWORDS: { word: string; weight: number }[] = [
  { word: 'kill', weight: 3 }, { word: 'murder', weight: 4 }, { word: 'massacre', weight: 5 },
  { word: 'slaughter', weight: 5 }, { word: 'bomb', weight: 4 }, { word: 'shoot', weight: 3 },
  { word: 'stab', weight: 3 }, { word: 'poison', weight: 3 }, { word: 'attack', weight: 2 },
  { word: 'destroy', weight: 2 }, { word: 'harm', weight: 2 }, { word: 'blast', weight: 3 },
  { word: 'explode', weight: 4 }, { word: 'detonate', weight: 5 }, { word: 'assault', weight: 3 },
  { word: 'torture', weight: 5 }, { word: 'execute', weight: 4 }, { word: 'behead', weight: 5 },
  { word: 'strangle', weight: 4 }, { word: 'suffocate', weight: 4 }, { word: 'maim', weight: 4 },
  { word: 'dismember', weight: 5 }, { word: 'annihilate', weight: 4 },
  { word: 'ram', weight: 4 }, { word: 'mow down', weight: 5 }, { word: 'run over', weight: 3 },
  { word: 'plow into', weight: 4 }, { word: 'drive into', weight: 3 },
];

// ── Intent patterns (more specific) ──
const INTENT_PATTERNS = [
  /\b(i|we)\s+(will|am\s+going\s+to|plan\s+to|intend\s+to|gonna)\b.{0,100}\b(kill|attack|bomb|shoot|stab|poison|destroy|explode|detonate|massacre|murder|torture|execute|ram|mow\s+down|plow|drive\s+into|run\s+over)\b/i,
  /\b(i|we)\s+(want|need|have)\s+to\b.{0,100}\b(kill|attack|bomb|shoot|stab|destroy|murder|torture|ram|mow\s+down)\b/i,
  /\b(i|we)\s+(bought|acquired|obtained|have)\b.{0,80}\b(gun|weapon|knife|explosive|bomb|rifle|pistol|ammunition|ammo)\b/i,
  /\b(target|targeting)\b.{0,60}\b(school|church|mosque|synagogue|hospital|airport|government|police|military)\b/i,
  // Vehicular attack patterns
  /\b(i|we)\s+(will|am\s+going\s+to|plan\s+to|gonna)\b.{0,80}\b(ram|drive|plow|crash|mow)\b.{0,60}\b(into|through|over)\b.{0,60}\b(crowd|people|them|pedestrians|gathering|parade|market|protest)\b/i,
  // Purpose / mission / goal declarations of violence
  /\b(my|our)\s+(purpose|mission|goal|objective|aim|plan)\s+(is|was)\s+to\b.{0,100}\b(kill|attack|bomb|shoot|stab|destroy|explode|detonate|massacre|murder|torture|eliminate|annihilate)\b/i,
  // "carrying out attacks" / "carry out" + violence
  /\b(carry|carrying)\s+out\b.{0,60}\b(attacks?|bombings?|assassinations?|massacre|destruction|strikes?)\b/i,
  // "by way of" + armed/destructive methods
  /\b(by\s+way\s+of|by\s+means\s+of|through)\s+(armed|violent|destructive|military)\b/i,
];

const HOSTAGE_PATTERNS = [
  /\b(i\s+have|i\s+took|i\s+captured|i\s+kidnapped|i\s+am\s+holding).{0,50}(your|the|a).{0,50}(child|daughter|son|wife|husband|family|hostage)/i,
  /\b(kidnap|abduct|take\s+hostage|hold\s+hostage)\b/i,
  /\b(ransom|pay\s+up|or\s+they\s+die|or\s+(he|she)\s+dies)\b/i,
];

const EXTORTION_PATTERNS = [
  /\b(pay|transfer|send).{0,50}(money|funds|bitcoin|crypto|ransom).{0,50}(or\s+else|or\s+we\s+will|or\s+i\s+will)/i,
  /\b(we\s+)?(breached|hacked|compromised|encrypted).{0,50}(system|server|database|files|data).{0,80}(pay|ransom|bitcoin)/i,
  /\b(ddos|denial\s+of\s+service).{0,60}(unless|until|pay)/i,
  // Conditional threat / coercion patterns
  /\b(failure|fail|refuse|refusing)\s+to\s+(meet|comply|cooperate|respond|pay|follow).{0,60}(result|lead|cause|end)\s+in\b.{0,80}(release|leak|expose|publish|harm|consequences|action)/i,
  /\b(meet\s+our\s+demands|comply\s+with|cooperate\s+or)\b.{0,80}(release|leak|expose|publish|consequences|regret|suffer)/i,
  /\b(unless|until)\s+(you|they|your\s+organization).{0,60}(comply|cooperate|pay|meet|agree)\b/i,
  /\b(sensitive\s+information|private\s+data|confidential|personal\s+files).{0,60}(released?|leaked?|exposed?|published?|made\s+public)/i,
  /\b(we\s+have|in\s+possession\s+of).{0,60}(sensitive|private|confidential|personal).{0,40}(information|data|files|documents|records)/i,
];

const DOXING_PATTERNS = [
  /\b(publish|share|leak|expose|dox|doxx|release|post).{0,50}(your|his|her|their)\s+(address|location|phone|identity|photos?|nudes|personal\s+info|sensitive\s+info(rmation)?|private\s+info(rmation)?)/i,
  /\b(i\s+know\s+where\s+you\s+live|i\s+found\s+your\s+address|i\s+have\s+your\s+info)/i,
  /\b(release|leak|expose).{0,40}(sensitive|private|confidential).{0,40}(information|data).{0,40}(associated\s+with|linked\s+to|about)\s+(your|his|her|their)/i,
];

// ── Veiled / contextual threat indicators ──
// These individually are weak but combined signal a veiled threat
const VEILED_THREAT_SIGNALS: { pattern: RegExp; label: string; weight: number }[] = [
  // Expressions of finality / crossing a line
  { pattern: /\b(done\s+(being\s+patient|waiting|asking)|no\s+going\s+back|crossed\s+the\s+line|past\s+the\s+point)\b/i, label: 'finality', weight: 3 },
  { pattern: /\b(i'?m\s+(not\s+interested\s+in|done\s+with|tired\s+of)\s+(empty\s+)?words)/i, label: 'finality', weight: 2 },
  // Ominous warnings / they'll see
  { pattern: /\b(they('ll|\s+will)\s+(understand|see|know|regret|pay|learn)|you('ll|\s+will)\s+(see|regret|understand|know\s+soon))\b/i, label: 'implied_threat', weight: 4 },
  { pattern: /\b(soon\s+enough|when\s+the\s+time\s+comes|mark\s+my\s+words|remember\s+this)\b/i, label: 'implied_threat', weight: 3 },
  // Pushed to action / cornered
  { pattern: /\b(pushed\s+into\s+a\s+corner|backs?\s+against\s+the\s+wall|nothing\s+(left\s+)?to\s+lose|no\s+(other\s+)?(choice|option))\b/i, label: 'desperation', weight: 3 },
  { pattern: /\b(stop\s+asking\s+(for\s+)?change\s+and\s+start\s+demanding)/i, label: 'escalation', weight: 4 },
  // Reference to upcoming events/locations as targets
  { pattern: /\b(event|gathering|celebration|ceremony|rally)\b.{0,80}\b(coming\s+up|this\s+week(end)?|tomorrow|tonight|soon)\b/i, label: 'event_targeting', weight: 4 },
  { pattern: /\b(symbolic|represents?\s+everything|stands?\s+for)\b.{0,60}\b(wrong|corrupt|broken|hypocrisy|system)\b/i, label: 'symbolic_targeting', weight: 3 },
  // Disruption / something big
  { pattern: /\b(something\s+big\s+(happens|will\s+happen|is\s+coming|needs\s+to)|make\s+(them|people)\s+(notice|pay\s+attention|listen))\b/i, label: 'disruption_intent', weight: 4 },
  { pattern: /\b(disrupt|force\s+them\s+to\s+(look|see|notice)|shake\s+them|wake\s+them\s+up)\b/i, label: 'disruption_intent', weight: 3 },
  // "Some of us are serious" type language
  { pattern: /\b(some\s+of\s+us\s+are\s+serious|we('re|\s+are)\s+serious|i('m|\s+am)\s+serious\s+about\s+this)\b/i, label: 'seriousness', weight: 3 },
  // Grievance + action framing
  { pattern: /\b(i\s+keep\s+thinking\s+about|can't\s+stop\s+thinking\s+about|obsess)/i, label: 'fixation', weight: 3 },
  { pattern: /\b(patience\s+(is|was)|waited\s+long\s+enough|tired\s+of\s+pretending|not\s+so\s+sure\s+anymore)/i, label: 'grievance', weight: 2 },
];

const VEILED_THREAT_THRESHOLD = 12; // combined weight needed to flag as ambiguous
const VEILED_THREAT_HIGH_THRESHOLD = 18; // combined weight to flag as clear threat

// ── Benign context with weighted scoring ──
const BENIGN_CONTEXTS: { category: string; words: string[]; threshold: number }[] = [
  { category: 'food', words: ['chicken', 'duck', 'pig', 'cow', 'beef', 'pork', 'lamb', 'cook', 'dinner', 'recipe', 'meal', 'restaurant', 'kitchen', 'chef', 'bake', 'grill', 'oven', 'fry'], threshold: 2 },
  { category: 'games', words: ['game', 'gaming', 'enemy', 'monster', 'boss', 'quest', 'player', 'character', 'level', 'spawn', 'respawn', 'loot', 'raid', 'pvp', 'mmo', 'rpg', 'fps', 'gameplay'], threshold: 2 },
  { category: 'work', words: ['meeting', 'work', 'office', 'presentation', 'deadline', 'project', 'manager', 'colleague', 'email', 'client', 'report', 'schedule'], threshold: 2 },
  { category: 'medical', words: ['doctor', 'hospital', 'appointment', 'treatment', 'surgery', 'nurse', 'patient', 'diagnosis', 'medicine', 'therapy', 'clinic'], threshold: 2 },
  { category: 'news', words: ['according', 'reported', 'police', 'authorities', 'investigation', 'suspect', 'incident', 'witness', 'article', 'source', 'news', 'journalist'], threshold: 2 },
  { category: 'fiction', words: ['story', 'novel', 'movie', 'film', 'character', 'plot', 'scene', 'book', 'fiction', 'script', 'chapter', 'writing', 'wrote', 'author'], threshold: 2 },
  { category: 'sports', words: ['game', 'match', 'score', 'team', 'coach', 'player', 'season', 'championship', 'tournament', 'goal', 'win', 'loss', 'field', 'court'], threshold: 2 },
  { category: 'history', words: ['history', 'historical', 'century', 'war', 'battle', 'ancient', 'empire', 'civilization', 'period', 'era', 'documentary'], threshold: 2 },
];

// ── Threat amplifiers (raise confidence when present) ──
const AMPLIFIERS = [
  { pattern: /\b(today|tonight|tomorrow|this\s+week|right\s+now|soon)\b/i, label: 'temporal', boost: 8 },
  { pattern: /\b(school|church|mosque|synagogue|hospital|airport|mall|stadium|concert|crowd|pedestrians|gathering|parade|market|protest|white\s+house|congress|capitol|government|embassy|pentagon|parliament|senate|city|cities)\b/i, label: 'target', boost: 10 },
  { pattern: /\b(car|truck|vehicle|van|suv)\b.{0,30}\b(into|through|over)\b/i, label: 'vehicular_attack', boost: 12 },
  { pattern: /\b(gun|weapon|knife|explosive|bomb|rifle|pistol|ammunition|ammo|machete|sword|armed|firearm|grenade)\b/i, label: 'weapon', boost: 12 },
  { pattern: /\b(allah|jihad|infidel|crusade|holy\s+war|caliphate|martyr)\b/i, label: 'extremism', boost: 10 },
  { pattern: /\b(everyone|all\s+of\s+them|as\s+many\s+as|maximum\s+casualties|body\s+count)\b/i, label: 'mass_target', boost: 15 },
];

export type AnalysisConfidence = 'clear_threat' | 'clear_safe' | 'ambiguous';

export interface RuleAnalysisResult {
  scanResult: ScanResult;
  confidence_level: AnalysisConfidence;
}

export function analyzeTextRules(text: string, inputType: string = 'text'): RuleAnalysisResult {
  const lower = text.toLowerCase();

  // ── Step 1: Check for safe idioms first ──
  let sanitized = lower;
  for (const idiom of SAFE_IDIOMS) {
    sanitized = sanitized.replace(idiom, '___SAFE___');
  }

  // ── Step 2: Check benign context ──
  for (const ctx of BENIGN_CONTEXTS) {
    const count = ctx.words.filter((w) => lower.includes(w)).length;
    if (count >= ctx.threshold) {
      // Even in benign context, check for explicit intent patterns
      const hasExplicitIntent = INTENT_PATTERNS.some((p) => p.test(sanitized));
      if (!hasExplicitIntent) {
        return {
          scanResult: {
            prediction: 'non_threatening',
            confidence: Math.min(92, 75 + count * 5),
            method: 'rule_based',
            indicators: [`benign_context:${ctx.category}`],
            inputType,
            timestamp: new Date(),
            inputPreview: text.slice(0, 120),
          },
          confidence_level: 'clear_safe',
        };
      }
    }
  }

  // ── Step 3: Check high-confidence threat patterns ──
  for (const p of HOSTAGE_PATTERNS) {
    if (p.test(sanitized)) {
      const match = sanitized.match(p);
      if (match && !hasNearbyNegation(sanitized, match.index || 0)) {
        return {
          scanResult: {
            prediction: 'threatening',
            confidence: 97,
            method: 'rule_based',
            indicators: ['hostage_threat'],
            inputType,
            timestamp: new Date(),
            inputPreview: text.slice(0, 120),
          },
          confidence_level: 'clear_threat',
        };
      }
    }
  }

  for (const p of INTENT_PATTERNS) {
    if (p.test(sanitized)) {
      const match = sanitized.match(p);
      if (match && !hasNearbyNegation(sanitized, match.index || 0)) {
        let conf = 90;
        const indicators: string[] = ['intent'];
        for (const amp of AMPLIFIERS) {
          if (amp.pattern.test(sanitized)) {
            conf = Math.min(99, conf + amp.boost);
            indicators.push(amp.label);
          }
        }
        return {
          scanResult: {
            prediction: 'threatening',
            confidence: conf,
            method: 'rule_based',
            indicators,
            inputType,
            timestamp: new Date(),
            inputPreview: text.slice(0, 120),
          },
          confidence_level: 'clear_threat',
        };
      }
    }
  }

  for (const p of EXTORTION_PATTERNS) {
    if (p.test(sanitized)) {
      const match = sanitized.match(p);
      if (match && !hasNearbyNegation(sanitized, match.index || 0)) {
        return {
          scanResult: {
            prediction: 'threatening',
            confidence: 95,
            method: 'rule_based',
            indicators: ['extortion_threat'],
            inputType,
            timestamp: new Date(),
            inputPreview: text.slice(0, 120),
          },
          confidence_level: 'clear_threat',
        };
      }
    }
  }

  for (const p of DOXING_PATTERNS) {
    if (p.test(sanitized)) {
      const match = sanitized.match(p);
      if (match && !hasNearbyNegation(sanitized, match.index || 0)) {
        return {
          scanResult: {
            prediction: 'threatening',
            confidence: 92,
            method: 'rule_based',
            indicators: ['doxing_threat'],
            inputType,
            timestamp: new Date(),
            inputPreview: text.slice(0, 120),
          },
          confidence_level: 'clear_threat',
        };
      }
    }
  }

  // ── Step 4: Weighted keyword analysis ──
  let totalWeight = 0;
  const foundKeywords: string[] = [];
  for (const { word, weight } of VIOLENCE_KEYWORDS) {
    const regex = new RegExp(`\\b${word}\\b`, 'gi');
    let match;
    while ((match = regex.exec(sanitized)) !== null) {
      if (sanitized.slice(Math.max(0, match.index - 5), match.index + word.length + 5).includes('___SAFE___')) continue;
      if (hasNearbyNegation(sanitized, match.index)) continue;
      totalWeight += weight;
      if (!foundKeywords.includes(word)) foundKeywords.push(word);
    }
  }

  if (totalWeight > 0) {
    const personalPronouns = /\b(i|we|me|us|my|our)\b/i.test(sanitized);
    const indicators: string[] = ['violence_keyword'];
    if (personalPronouns) indicators.push('personal_pronoun');

    let ampBoost = 0;
    for (const amp of AMPLIFIERS) {
      if (amp.pattern.test(sanitized)) {
        ampBoost += amp.boost;
        indicators.push(amp.label);
      }
    }

    const baseScore = Math.min(70, 30 + totalWeight * 6);
    const pronounBonus = personalPronouns ? 12 : 0;
    const finalScore = Math.min(95, baseScore + pronounBonus + ampBoost);

    if (finalScore >= 80) {
      return {
        scanResult: {
          prediction: 'threatening',
          confidence: finalScore,
          method: 'keyword_heuristic',
          indicators,
          inputType,
          timestamp: new Date(),
          inputPreview: text.slice(0, 120),
        },
        confidence_level: finalScore >= 88 ? 'clear_threat' : 'ambiguous',
      };
    }

    if (finalScore >= 45) {
      return {
        scanResult: {
          prediction: totalWeight >= 6 ? 'threatening' : 'non_threatening',
          confidence: finalScore,
          method: 'keyword_heuristic',
          indicators,
          inputType,
          timestamp: new Date(),
          inputPreview: text.slice(0, 120),
        },
        confidence_level: 'ambiguous',
      };
    }
  }

  // ── Step 5: Check for veiled / contextual threats ──
  let veiledWeight = 0;
  const veiledIndicators: string[] = [];
  for (const signal of VEILED_THREAT_SIGNALS) {
    if (signal.pattern.test(sanitized)) {
      veiledWeight += signal.weight;
      if (!veiledIndicators.includes(signal.label)) {
        veiledIndicators.push(signal.label);
      }
    }
  }

  if (veiledWeight >= VEILED_THREAT_HIGH_THRESHOLD) {
    return {
      scanResult: {
        prediction: 'threatening',
        confidence: Math.min(92, 70 + veiledWeight),
        method: 'rule_based',
        indicators: veiledIndicators,
        inputType,
        timestamp: new Date(),
        inputPreview: text.slice(0, 120),
      },
      confidence_level: 'ambiguous',
    };
  }

  if (veiledWeight >= VEILED_THREAT_THRESHOLD) {
    return {
      scanResult: {
        prediction: 'non_threatening',
        confidence: 50,
        method: 'rule_based',
        indicators: veiledIndicators,
        inputType,
        timestamp: new Date(),
        inputPreview: text.slice(0, 120),
      },
      confidence_level: 'ambiguous',
    };
  }

  // ── Step 6: No signals found ──
  if (text.length > 200) {
    return {
      scanResult: {
        prediction: 'non_threatening',
        confidence: 65,
        method: 'rule_based',
        indicators: [],
        inputType,
        timestamp: new Date(),
        inputPreview: text.slice(0, 120),
      },
      confidence_level: 'ambiguous',
    };
  }

  return {
    scanResult: {
      prediction: 'non_threatening',
      confidence: 88,
      method: 'rule_based',
      indicators: [],
      inputType,
      timestamp: new Date(),
      inputPreview: text.slice(0, 120),
    },
    confidence_level: 'clear_safe',
  };
}
