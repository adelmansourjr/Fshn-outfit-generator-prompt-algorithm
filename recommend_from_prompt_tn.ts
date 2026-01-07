#!/usr/bin/env ts-node

/**
 * FSHN outfit recommender with tensor-network style scoring + Gemini 2.5 Flash.
 *
 * Pipeline:
 *  - Load index.json from classifier (includes: category, colours, vibes, gender, fit, sportMeta, entities).
 *  - Parse prompt with light heuristics.
 *  - Ask Gemini 2.5 Flash for an "intent" JSON (mode, form, vibes, colours, sport/team/brand, fit).
 *  - Construct prompt weight vectors (w_colour, w_vibe, w_fit, w_sport, w_brand, w_team).
 *  - For each required category, filter candidate items (gender, sport/team/brand).
 *  - Score single items with TN-style alignment: colour + harmony, vibe, fit, brand, sport, team, specific-name match.
 *  - If mode=outfit, assemble outfits across roles and apply pairwise compatibility
 *    (colour harmony, vibe harmony, fit silhouette, sport/team coherence).
 *  - Sample with jitter + epsilon so results vary but stay on-theme.
 *
 * Output format:
 *  - Outfit: "category imagePath" one per line, in role order, separate outfits with a blank line.
 *  - Single: same, but only requested categories.
 */

import fs from 'fs';
import path from 'path';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { VertexAI } from '@google-cloud/vertexai';

type CategoryMain = 'top' | 'bottom' | 'shoes' | 'mono';
type Gender = 'men' | 'women' | 'unisex';
type Colour =
  | 'black'
  | 'white'
  | 'grey'
  | 'red'
  | 'blue'
  | 'green'
  | 'beige'
  | 'brown'
  | 'pink'
  | 'yellow'
  | 'purple';
type Vibe =
  | 'streetwear'
  | 'edgy'
  | 'minimal'
  | 'y2k'
  | 'techwear'
  | 'sporty'
  | 'preppy'
  | 'vintage'
  | 'chic';
type Fit = 'oversized' | 'regular' | 'slim' | 'cropped';
type Sport = 'football' | 'basketball' | 'running' | 'tennis' | 'gym' | 'other' | 'none';

type EntityType = 'brand' | 'team' | 'sponsor' | 'generic';

interface EntityMeta {
  text: string;
  weight: number;
  type: EntityType;
}

interface SportMeta {
  sport?: Sport | null;
  teams?: string[];
  isKit?: boolean;
}

interface IndexItem {
  id: string;
  imagePath: string;
  category: CategoryMain;
  sub?: string | null;
  colours: Colour[];
  vibes: Vibe[];
  gender: Gender;
  fit?: Fit | null;
  sportMeta?: SportMeta | null;
  name?: string | null;
  name_normalized?: string | null;
  entities?: string[];
  entityMeta?: EntityMeta[];
}

// Intent schema we expect from Gemini (and fall back to if needed)
type Mode = 'outfit' | 'single';
type OutfitForm =
  | 'top_bottom_shoes'
  | 'top_bottom'
  | 'mono_only'
  | 'mono_and_shoes'
  | 'top_only'
  | 'bottom_only'
  | 'shoes_only';

interface PromptIntent {
  outfit_mode: Mode;
  requested_form: OutfitForm;
  required_categories: CategoryMain[];
  optional_categories: CategoryMain[];
  target_gender: Gender | 'any';
  vibe_tags: Vibe[];
  colour_hints: Colour[];
  brand_focus: string[];
  team_focus: string[];
  sport_context: Sport;
  fit_preference: Fit | 'mixed' | null;
  specific_items: string[];
}

// CLI
const argv = yargs(hideBin(process.argv))
  .option('index', {
    type: 'string',
    demandOption: true,
    describe: 'Path to index.json produced by classifier',
  })
  .option('prompt', {
    type: 'string',
    demandOption: true,
    describe: 'User text prompt, e.g. "playboi carti fit with timberlands"',
  })
  .option('gender_pref', {
    type: 'string',
    default: 'any',
    choices: ['any', 'men', 'women'],
  })
  .option('project', {
    type: 'string',
    describe: 'Google Cloud project ID (for Vertex)',
    demandOption: true,
  })
  .option('location', {
    type: 'string',
    default: 'us-east5',
    describe: 'Vertex AI location',
  })
  .option('model', {
    type: 'string',
    default: 'gemini-2.5-flash',
    describe: 'Gemini model name',
  })
  .option('pool_size', {
    type: 'number',
    default: 6,
    describe: 'Number of outfits/items to output',
  })
  .option('per_role_limit', {
    type: 'number',
    default: 12,
    describe: 'Max candidates per role before combinatorics',
  })
  .option('epsilon', {
    type: 'number',
    default: 0.15,
    describe: 'Diversity mass mixing factor (0–0.5)',
  })
  .option('jitter', {
    type: 'number',
    default: 0.15,
    describe: 'Uniform jitter range added to scores',
  })
  .option('debug', {
    type: 'boolean',
    default: false,
  })
  .parseSync();

const DEBUG = !!argv.debug || !!process.env.DEBUG_OUTFITS;

// Allowed vocab
const ALLOWED = {
  colours: [
    'black',
    'white',
    'grey',
    'red',
    'blue',
    'green',
    'beige',
    'brown',
    'pink',
    'yellow',
    'purple',
  ] as const,
  vibes: [
    'streetwear',
    'edgy',
    'minimal',
    'y2k',
    'techwear',
    'sporty',
    'preppy',
    'vintage',
    'chic',
  ] as const,
};

// ------------------------- Small helpers -------------------------

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
function randUniform(min = 0, max = 1): number {
  return min + Math.random() * (max - min);
}
function choiceWeighted<T>(items: T[], scores: number[]): T | null {
  const total = scores.reduce((a, b) => a + b, 0);
  if (!items.length || total <= 0) return null;
  let r = Math.random() * total;
  for (let i = 0; i < items.length; i++) {
    r -= scores[i];
    if (r <= 0) return items[i];
  }
  return items[items.length - 1] ?? null;
}

function toColour(s: string): Colour | null {
  const t = s.toLowerCase().trim() as Colour;
  return (ALLOWED.colours as readonly string[]).includes(t) ? t : null;
}
function toVibe(s: string): Vibe | null {
  const t = s.toLowerCase().trim() as Vibe;
  return (ALLOWED.vibes as readonly string[]).includes(t) ? t : null;
}
function normalizeText(s: string | null | undefined): string {
  return (s || '')
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function logDebug(...args: any[]) {
  if (DEBUG) console.error('[DEBUG]', ...args);
}

// ------------------------- Gemini Intent -------------------------

async function getIntentFromGemini(
  prompt: string,
  genderPref: 'any' | 'men' | 'women',
): Promise<PromptIntent | null> {
  const vertex = new VertexAI({
    project: argv.project as string,
    location: argv.location as string,
  });

  const systemPrompt = `
You are an intent parser for a fashion outfit recommender.
Your ONLY job is to convert a user prompt into a structured JSON object.

- Do NOT return explanations, markdown, code fences, or comments.
- Return a SINGLE JSON object only.
- JSON MUST match this TypeScript-like schema:

type CategoryMain = "top" | "bottom" | "shoes" | "mono";
type Gender = "men" | "women" | "unisex";
type Colour = "black"|"white"|"grey"|"red"|"blue"|"green"|"beige"|"brown"|"pink"|"yellow"|"purple";
type Vibe = "streetwear"|"edgy"|"minimal"|"y2k"|"techwear"|"sporty"|"preppy"|"vintage"|"chic";
type Fit = "oversized"|"regular"|"slim"|"cropped";
type Sport = "football"|"basketball"|"running"|"tennis"|"gym"|"other"|"none";

interface PromptIntent {
  outfit_mode: "outfit" | "single";
  requested_form:
    | "top_bottom_shoes"
    | "top_bottom"
    | "mono_only"
    | "mono_and_shoes"
    | "top_only"
    | "bottom_only"
    | "shoes_only";
  required_categories: CategoryMain[];
  optional_categories: CategoryMain[];
  target_gender: Gender | "any";
  vibe_tags: Vibe[];
  colour_hints: Colour[];
  brand_focus: string[];  // brand words explicitly mentioned in the prompt
  team_focus: string[];   // sports team names explicitly mentioned in the prompt
  sport_context: Sport;
  fit_preference: Fit | "mixed" | null;
  specific_items: string[]; // item names the user clearly refers to, exactly as text
}

Rules:
- If user says "fit", "outfit", "look", or mentions multiple categories (top+bottom etc.), use outfit_mode="outfit".
- If user clearly asks for only one category ("just shoes", "hoodie only"), use outfit_mode="single" and adjust requested_form.
- brand_focus: collect brand tokens mentioned in the text, in lowercase (e.g. "yeezy", "timberland", "zara", "nike").
- team_focus: collect any sports teams or clubs mentioned (football clubs, NBA teams, etc.), as lowercase strings.
- sport_context:
  - "football" if they clearly talk about football / soccer kits, matches, or football teams.
  - "basketball" if they clearly talk about basketball or NBA jerseys/teams.
  - "gym" if it is clearly workout/gym/training clothing.
  - "running" or "tennis" when explicitly mentioned.
  - "none" when the prompt is just fashion with no explicit sport.
- vibe_tags:
  - "streetwear" for hoodies, cargos, graphic tees, hype sneakers, rap/hip-hop adjacent fits.
  - "edgy" for black, leather, punk, goth, grunge.
  - "minimal" for clean basics, essentials, neutral tones, capsule wardrobes.
  - "sporty" for sportswear, jerseys, track pants, clearly athletic looks.
  - Others as they naturally fit (y2k, preppy, vintage, chic).
  - Use up to 2–3 that best match the prompt.
- fit_preference:
  - "oversized" if they say baggy, oversized, huge, boxy, etc.
  - "slim" if they say skinny, tight, fitted.
  - "cropped" for cropped tops, shorter jackets.
  - "regular" when neither oversized nor slim is emphasised.
  - "mixed" if they explicitly want a mix (e.g. oversized top with slim bottoms).
- specific_items:
  - Include phrases like "yeezy dove hoodie", "timberland boots", "black leather biker jacket" when the user clearly points to particular pieces.
  - Use natural text chunks from the prompt.

Return ONLY the JSON. No backticks, no leading text.
`.trim();

  const userPrompt = `User prompt: "${prompt}"
Gender preference hint: "${genderPref}"`;

  const model = vertex.getGenerativeModel({
    model: argv.model as string,
    systemInstruction: {
      role: 'system',
      parts: [{ text: systemPrompt }],
    },
  });

  const result = await model.generateContent({
    contents: [
      {
        role: 'user',
        parts: [{ text: userPrompt }],
      },
    ],
  });

  const text =
    result.response?.candidates?.[0]?.content?.parts?.[0]?.text ?? '';

  if (!text.trim()) {
    logDebug('Empty Gemini response for intent');
    return null;
  }

  try {
    const firstBrace = text.indexOf('{');
    const lastBrace = text.lastIndexOf('}');
    if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
      throw new Error('No JSON braces found');
    }
    const jsonStr = text.slice(firstBrace, lastBrace + 1);
    const parsed = JSON.parse(jsonStr) as PromptIntent;

    // Light sanitisation
    parsed.required_categories = (parsed.required_categories || []).filter(
      (c) => ['top', 'bottom', 'shoes', 'mono'].includes(c),
    );
    parsed.optional_categories = (parsed.optional_categories || []).filter(
      (c) => ['top', 'bottom', 'shoes', 'mono'].includes(c),
    );
    parsed.vibe_tags = (parsed.vibe_tags || [])
      .map(toVibe)
      .filter((v): v is Vibe => !!v);
    parsed.colour_hints = (parsed.colour_hints || [])
      .map(toColour)
      .filter((c): c is Colour => !!c);

    return parsed;
  } catch (err) {
    logDebug('Failed to parse JSON from Gemini. Raw response:\n', text);
    logDebug(err);
    return null;
  }
}

// ------------------------- Heuristic fallback intent -------------------------

function guessIntentHeuristic(
  prompt: string,
  genderPref: 'any' | 'men' | 'women',
): PromptIntent {
  const p = normalizeText(prompt);

  const has = (...ks: string[]) => ks.some((k) => p.includes(k));

  const outfitMode: Mode =
    has('fit', 'outfit', 'look') || has('top and bottom', 'top bottom')
      ? 'outfit'
      : 'single';

  let requested_form: OutfitForm = 'top_bottom_shoes';
  if (outfitMode === 'single') {
    if (has('shoes', 'boots', 'sneakers', 'trainers')) requested_form = 'shoes_only';
    else if (has('pants', 'jeans', 'trousers', 'bottom')) requested_form = 'bottom_only';
    else requested_form = 'top_only';
  } else {
    if (has('dress', 'gown', 'slip dress')) requested_form = 'mono_only';
  }

  const required_categories: CategoryMain[] = [];
  if (requested_form === 'top_bottom_shoes') {
    required_categories.push('top', 'bottom', 'shoes');
  } else if (requested_form === 'mono_only') {
    required_categories.push('mono');
  } else if (requested_form === 'top_only') {
    required_categories.push('top');
  } else if (requested_form === 'bottom_only') {
    required_categories.push('bottom');
  } else if (requested_form === 'shoes_only') {
    required_categories.push('shoes');
  } else if (requested_form === 'mono_and_shoes') {
    required_categories.push('mono', 'shoes');
  }

  // sport_context
  let sport_context: Sport = 'none';
  if (has('football', 'soccer', 'kit', 'matchday')) {
    sport_context = 'football';
  } else if (has('basketball', 'nba')) {
    sport_context = 'basketball';
  } else if (has('gym', 'workout', 'training')) {
    sport_context = 'gym';
  } else if (has('running', 'runner')) {
    sport_context = 'running';
  } else if (has('tennis')) {
    sport_context = 'tennis';
  }

  // vibes
  const vibe_tags: Vibe[] = [];
  if (has('streetwear', 'street', 'trap', 'hoodie', 'cargo', 'graphic', 'hype')) {
    vibe_tags.push('streetwear');
  }
  if (has('grunge', 'goth', 'rock', 'punk', 'dark', 'leather')) {
    vibe_tags.push('edgy');
  }
  if (has('minimal', 'clean', 'basic', 'essentials', 'capsule', 'neutral')) {
    vibe_tags.push('minimal');
  }
  if (has('y2k', '2000')) vibe_tags.push('y2k');
  if (has('preppy', 'college', 'ivy', 'varsity', 'polo')) {
    vibe_tags.push('preppy');
  }
  if (has('vintage', 'retro', 'oldschool')) {
    vibe_tags.push('vintage');
  }
  if (has('chic', 'elegant', 'slip dress', 'silk', 'satin')) {
    vibe_tags.push('chic');
  }
  if (sport_context !== 'none') vibe_tags.push('sporty');

  // fit
  let fit_preference: Fit | 'mixed' | null = null;
  if (has('baggy', 'oversized', 'huge', 'boxy')) {
    fit_preference = 'oversized';
  } else if (has('skinny', 'tight', 'slim')) {
    fit_preference = 'slim';
  } else if (has('cropped')) {
    fit_preference = 'cropped';
  }

  return {
    outfit_mode: outfitMode,
    requested_form,
    required_categories,
    optional_categories: [],
    target_gender: genderPref === 'any' ? 'any' : (genderPref as Gender),
    vibe_tags: Array.from(new Set(vibe_tags)).slice(0, 3),
    colour_hints: [],
    brand_focus: [],
    team_focus: [],
    sport_context,
    fit_preference: fit_preference ?? 'mixed',
    specific_items: [],
  };
}

// ------------------------- Weight construction -------------------------

interface ContextWeights {
  wColour: Record<Colour, number>;
  wVibe: Record<Vibe, number>;
  wFit: Record<Fit, number>;
  wSport: Record<Sport, number>;
  brandTokens: string[];
  teamTokens: string[];
  sportContext: Sport;
  specificTokens: string[];
}

function buildWeights(intent: PromptIntent): ContextWeights {
  const wColour: Record<Colour, number> = {
    black: 0,
    white: 0,
    grey: 0,
    red: 0,
    blue: 0,
    green: 0,
    beige: 0,
    brown: 0,
    pink: 0,
    yellow: 0,
    purple: 0,
  };
  const wVibe: Record<Vibe, number> = {
    streetwear: 0,
    edgy: 0,
    minimal: 0,
    y2k: 0,
    techwear: 0,
    sporty: 0,
    preppy: 0,
    vintage: 0,
    chic: 0,
  };
  const wFit: Record<Fit, number> = {
    oversized: 0.2,
    regular: 0.4,
    slim: 0.2,
    cropped: 0.2,
  };
  const sports: Sport[] = [
    'none',
    'football',
    'basketball',
    'running',
    'tennis',
    'gym',
    'other',
  ];
  const wSport: Record<Sport, number> = {
    none: 0,
    football: 0,
    basketball: 0,
    running: 0,
    tennis: 0,
    gym: 0,
    other: 0,
  };

  // Colours: hint colours get 1.0, neutrals get a small baseline if user didn't specify colours.
  for (const c of intent.colour_hints || []) {
    if (c in wColour) wColour[c] += 1.0;
  }
  if ((intent.colour_hints || []).length === 0) {
    wColour.black += 0.3;
    wColour.white += 0.3;
    wColour.grey += 0.3;
    wColour.beige += 0.2;
  }

  // Vibes from intent
  for (const v of intent.vibe_tags || []) {
    if (v in wVibe) wVibe[v] += 1.0;
  }

  // Fit
  if (intent.fit_preference && intent.fit_preference !== 'mixed') {
    for (const f of Object.keys(wFit) as Fit[]) {
      wFit[f] = 0.1;
    }
    wFit[intent.fit_preference] = 1.0;
  } else if (!intent.fit_preference || intent.fit_preference === 'mixed') {
    // slight bias: oversized + slim + regular; exact silhouette is handled in pairwise fit
    wFit.oversized += 0.2;
    wFit.slim += 0.2;
    wFit.regular += 0.5;
  }

  // Sport
  if (intent.sport_context && intent.sport_context !== 'none') {
    for (const s of sports) wSport[s] = 0;
    wSport[intent.sport_context] = 1.0;
  }

  const brandTokens = (intent.brand_focus || []).map(normalizeText);
  const teamTokens = (intent.team_focus || []).map(normalizeText);

  // Specific item tokens (e.g. "yeezy dove hoodie", "timberland boots")
  const specificTokensSet = new Set<string>();
  for (const raw of intent.specific_items || []) {
    const norm = normalizeText(raw);
    if (!norm) continue;
    for (const w of norm.split(' ')) {
      if (w) specificTokensSet.add(w);
    }
  }
  const specificTokens = Array.from(specificTokensSet);

  return {
    wColour,
    wVibe,
    wFit,
    wSport,
    brandTokens,
    teamTokens,
    sportContext: intent.sport_context,
    specificTokens,
  };
}

// ------------------------- Single item scoring -------------------------

const WEIGHT_SINGLE = {
  color: 1.0,
  vibe: 1.2,
  fit: 0.8,
  brand: 1.0,
  sport: 1.0,
  team: 1.2,
  specific: 1.5, // stronger pull toward explicitly named items (e.g. "timberland boots")
};

function colourAlignment(item: IndexItem, wColour: Record<Colour, number>): number {
  if (!item.colours || item.colours.length === 0) return 0;

  let best = 0;
  for (const c of item.colours) {
    best = Math.max(best, wColour[c] || 0);
  }

  const totalPref = Object.values(wColour).reduce((a, b) => a + b, 0);
  const hasMatch = item.colours.some((c) => (wColour[c] || 0) > 0);
  if (totalPref > 0 && !hasMatch) {
    // If the user expressed clear colour preferences and this item has none of them,
    // apply a penalty so off-tone pieces (e.g. bright kits) drop down.
    best -= 0.5;
  }

  // neutrals harmony: if item has neutrals and any neutral has weight, small bonus
  const neutrals: Colour[] = ['black', 'white', 'grey', 'beige', 'brown'];
  const hasNeutral = item.colours.some((c) => neutrals.includes(c));
  if (hasNeutral) {
    const neutralWeight = Math.max(...neutrals.map((c) => wColour[c] || 0));
    best += 0.2 * neutralWeight;
  }
  return best;
}

function vibeAlignment(item: IndexItem, wVibe: Record<Vibe, number>): number {
  if (!item.vibes || item.vibes.length === 0) return 0;

  let score = 0;
  for (const v of item.vibes) {
    score += wVibe[v] || 0;
  }

  const totalPref = Object.values(wVibe).reduce((a, b) => a + b, 0);
  const hasMatch = item.vibes.some((v) => (wVibe[v] || 0) > 0);
  if (totalPref > 0 && !hasMatch) {
    // If the user has explicit vibe preferences and this item doesn't match any of them,
    // nudge it down so pure "sporty" pieces don't dominate a non-sport prompt.
    score -= 0.3;
  }

  return score;
}

function fitAlignment(item: IndexItem, wFit: Record<Fit, number>): number {
  const f = (item.fit || 'regular') as Fit;
  return wFit[f] || 0;
}

function brandAlignment(item: IndexItem, brandTokens: string[]): number {
  if (!brandTokens.length) return 0;
  const nameText =
    normalizeText(item.name || '') + ' ' + normalizeText(item.name_normalized || '');
  const entText = normalizeText(
    (item.entityMeta || []).map((e) => e.text).join(' '),
  );

  let score = 0;
  for (const t of brandTokens) {
    if (!t) continue;
    if (nameText.includes(t)) score += 1.0;
    if (entText.includes(t)) score += 1.5;
  }
  return score;
}

function sportAlignment(
  item: IndexItem,
  wSport: Record<Sport, number>,
  sportContext: Sport,
): number {
  const sm = item.sportMeta || {};
  const sport = (sm.sport || 'none') as Sport;
  let score = wSport[sport] || 0;

  // If the intent is explicitly sporty, reward matching kits slightly.
  if ((wSport[sport] || 0) > 0 && sm.isKit) score += 0.5;

  // If the user *isn't* asking for a sport fit, push explicit sport items down
  // so football shorts/jerseys don't leak into neutral streetwear prompts.
  if (sportContext === 'none' && sport !== 'none') {
    score -= sm.isKit ? 0.7 : 0.4;
  }

  return score;
}

function specificItemAlignment(
  item: IndexItem,
  specificTokens: string[],
): number {
  if (!specificTokens.length) return 0;
  const txt =
    normalizeText(item.name || '') +
    ' ' +
    normalizeText(item.name_normalized || '') +
    ' ' +
    normalizeText(item.imagePath || '') +
    ' ' +
    normalizeText((item.entityMeta || []).map((e) => e.text).join(' '));

  let score = 0;
  for (const t of specificTokens) {
    if (!t) continue;
    if (txt.includes(t)) score += 1.0;
  }
  return score;
}

function countSpecificMatches(
  item: IndexItem,
  specificTokens: string[],
): number {
  if (!specificTokens.length) return 0;
  const txt =
    normalizeText(item.name || '') +
    ' ' +
    normalizeText(item.name_normalized || '') +
    ' ' +
    normalizeText(item.imagePath || '') +
    ' ' +
    normalizeText((item.entityMeta || []).map((e) => e.text).join(' '));
  let count = 0;
  for (const t of specificTokens) {
    if (!t) continue;
    if (txt.includes(t)) count++;
  }
  return count;
}

function genericTeamText(item: IndexItem): string {
  return (
    normalizeText(item.name || '') +
    ' ' +
    normalizeText(item.name_normalized || '') +
    ' ' +
    normalizeText(item.imagePath || '') +
    ' ' +
    normalizeText((item.entities || []).join(' ')) +
    ' ' +
    normalizeText((item.entityMeta || []).map((e) => e.text).join(' '))
  );
}

function teamAlignment(
  item: IndexItem,
  teamTokens: string[],
): number {
  if (!teamTokens.length) return 0;
  const teams = (item.sportMeta?.teams || []).map(normalizeText);
  const entTeams = (item.entityMeta || [])
    .filter((e) => e.type === 'team')
    .map((e) => normalizeText(e.text));
  const txt = genericTeamText(item);

  let score = 0;
  for (const t of teamTokens) {
    if (!t) continue;
    if (teams.some((x) => x && (x.includes(t) || t.includes(x)))) score += 1.5;
    if (entTeams.some((x) => x && (x.includes(t) || t.includes(x)))) score += 1.5;
    if (txt.includes(t)) score += 1.0;
  }
  return score;
}

function scoreSingleItem(
  item: IndexItem,
  weights: ContextWeights,
): number {
  const c = colourAlignment(item, weights.wColour);
  const v = vibeAlignment(item, weights.wVibe);
  const f = fitAlignment(item, weights.wFit);
  const b = brandAlignment(item, weights.brandTokens);
  const s = sportAlignment(item, weights.wSport, weights.sportContext);
  const t = teamAlignment(item, weights.teamTokens);
  const sp = specificItemAlignment(item, weights.specificTokens);

  const score =
    WEIGHT_SINGLE.color * c +
    WEIGHT_SINGLE.vibe * v +
    WEIGHT_SINGLE.fit * f +
    WEIGHT_SINGLE.brand * b +
    WEIGHT_SINGLE.sport * s +
    WEIGHT_SINGLE.team * t +
    WEIGHT_SINGLE.specific * sp;

  return score;
}

// ------------------------- Pairwise compatibility (TN factors) -------------------------

const WEIGHT_PAIR = {
  color: 0.6,
  vibe: 0.9,
  fit: 0.7,
  sport: 0.8,
  team: 1.0,
};

function pairColorCompatibility(a: IndexItem, b: IndexItem): number {
  if (!a.colours.length || !b.colours.length) return 0;
  let score = 0;
  for (const ca of a.colours) {
    for (const cb of b.colours) {
      if (ca === cb) {
        score += 1.0;
      } else {
        const neutrals: Colour[] = ['black', 'white', 'grey', 'beige', 'brown'];
        const isNeutralA = neutrals.includes(ca);
        const isNeutralB = neutrals.includes(cb);
        if (isNeutralA && isNeutralB) score += 0.6;
        else if (isNeutralA || isNeutralB) score += 0.4;
      }
    }
  }
  return score / (a.colours.length * b.colours.length);
}

function pairVibeCompatibility(a: IndexItem, b: IndexItem): number {
  const va = new Set(a.vibes || []);
  const vb = new Set(b.vibes || []);
  let shared = 0;
  for (const v of va) if (vb.has(v)) shared++;

  let bonus = 0;
  if (va.has('streetwear') && vb.has('sporty')) bonus += 0.5;
  if (va.has('sporty') && vb.has('streetwear')) bonus += 0.5;
  if (va.has('chic') && vb.has('minimal')) bonus += 0.4;
  if (va.has('minimal') && vb.has('chic')) bonus += 0.4;

  return shared + bonus;
}

function pairFitCompatibility(top: IndexItem, bottom: IndexItem): number {
  const ft = (top.fit || 'regular') as Fit;
  const fb = (bottom.fit || 'regular') as Fit;
  if (ft === 'oversized' && fb === 'slim') return 1.3;
  if (ft === 'oversized' && fb === 'regular') return 1.0;
  if (ft === 'regular' && fb === 'slim') return 1.0;
  if (ft === 'slim' && fb === 'slim') return 0.9;
  if (ft === 'oversized' && fb === 'oversized') return 0.4;
  return 0.5;
}

function pairSportCompatibility(
  a: IndexItem,
  b: IndexItem,
  sportContext: Sport,
): number {
  const sa = (a.sportMeta?.sport || 'none') as Sport;
  const sb = (b.sportMeta?.sport || 'none') as Sport;

  if (sportContext === 'none') {
    if (sa === 'none' && sb === 'none') return 0;
    if (sa !== 'none' && sb !== 'none') return -0.5; // two sport pieces in a non-sport fit
    return -0.2; // one sport piece mixed into a non-sport outfit
  }

  if (sa === 'none' || sb === 'none') return 0;
  if (sa === sb) return 1.0;
  return 0.2;
}

function pairTeamCompatibility(a: IndexItem, b: IndexItem): number {
  const ta = (a.sportMeta?.teams || []).map(normalizeText);
  const tb = (b.sportMeta?.teams || []).map(normalizeText);
  if (!ta.length || !tb.length) return 0;
  for (const x of ta) {
    for (const y of tb) {
      if (x && y && (x === y || x.includes(y) || y.includes(x))) {
        return 1.2;
      }
    }
  }
  return -0.3; // slight penalty for mixed teams in sport contexts
}

// ------------------------- Candidate filtering -------------------------

function genderCompatible(
  itemGender: Gender,
  target: Gender | 'any',
): boolean {
  if (target === 'any') return true;
  if (itemGender === 'unisex') return true;
  return itemGender === target;
}

function hasBrandMatch(item: IndexItem, brandTokens: string[]): boolean {
  if (!brandTokens.length) return false;
  const txt =
    normalizeText(item.name || '') +
    ' ' +
    normalizeText(item.name_normalized || '') +
    ' ' +
    normalizeText((item.entityMeta || []).map((e) => e.text).join(' '));
  return brandTokens.some((b) => b && txt.includes(b));
}

function hasTeamMatch(item: IndexItem, teamTokens: string[]): boolean {
  if (!teamTokens.length) return false;
  const teams = (item.sportMeta?.teams || []).map(normalizeText);
  const entTeams = (item.entityMeta || [])
    .filter((e) => e.type === 'team')
    .map((e) => normalizeText(e.text));
  const all = teams.concat(entTeams);
  const txt = genericTeamText(item);
  return teamTokens.some(
    (t) =>
      t &&
      (all.some((x) => x && (x.includes(t) || t.includes(x))) ||
        txt.includes(t)),
  );
}

function sportMatches(item: IndexItem, sport: Sport): boolean {
  if (!sport || sport === 'none') return true;
  return (item.sportMeta?.sport || 'none') === sport;
}

function isSportRelevantForContext(
  item: IndexItem,
  sportContext: Sport,
  teamTokens: string[],
): boolean {
  if (sportContext === 'none') return false;
  const sport = (item.sportMeta?.sport || 'none') as Sport;
  if (sport === sportContext) return true;
  if (hasTeamMatch(item, teamTokens)) return true;
  return false;
}

function filterCandidatesForRole(
  items: IndexItem[],
  category: CategoryMain,
  intent: PromptIntent,
  weights: ContextWeights,
): IndexItem[] {
  let base = items.filter((it) => it.category === category);

  // Strong sport-aware narrowing: if this is a sport look and we can identify
  // sport-relevant items for this role (e.g. Barça shorts), drop everything else.
  if (intent.sport_context !== 'none') {
    const sporty = base.filter((it) =>
      isSportRelevantForContext(it, intent.sport_context, weights.teamTokens),
    );
    if (sporty.length) {
      base = sporty;
    }
  }

  const stages: {
    requireBrand: boolean;
    requireTeam: boolean;
    requireSportStrong: boolean;
  }[] = [
    { requireBrand: true, requireTeam: true, requireSportStrong: true },
    { requireBrand: false, requireTeam: true, requireSportStrong: true },
    { requireBrand: false, requireTeam: false, requireSportStrong: true },
    { requireBrand: false, requireTeam: false, requireSportStrong: false },
  ];

  for (const stage of stages) {
    let pool = base.filter((it) =>
      genderCompatible(it.gender, intent.target_gender),
    );

    if (stage.requireSportStrong) {
      if (intent.sport_context !== 'none') {
        pool = pool.filter((it) =>
          isSportRelevantForContext(it, intent.sport_context, weights.teamTokens),
        );
      } else {
        // Non-sport look: prefer non-sport items when possible.
        const nonSport = pool.filter(
          (it) => (it.sportMeta?.sport || 'none') === 'none',
        );
        if (nonSport.length) pool = nonSport;
      }
    }

    if (stage.requireBrand && weights.brandTokens.length) {
      pool = pool.filter((it) => hasBrandMatch(it, weights.brandTokens));
    }

    if (stage.requireTeam && weights.teamTokens.length) {
      pool = pool.filter((it) => hasTeamMatch(it, weights.teamTokens));
    }

    // If user named specific items, prefer items in this category that match the
    // highest number of those tokens. This pulls in "timberland boots" etc.
    if (weights.specificTokens.length && pool.length) {
      const counted = pool.map((it) => ({
        item: it,
        count: countSpecificMatches(it, weights.specificTokens),
      }));
      const maxMatches = Math.max(...counted.map((c) => c.count));
      if (maxMatches > 0) {
        pool = counted
          .filter((c) => c.count === maxMatches)
          .map((c) => c.item);
      }
    }

    if (pool.length) return pool;
  }

  // As absolute fallback, return all items of that category
  return base;
}

// ------------------------- Outfit assembly -------------------------

interface Outfit {
  top?: IndexItem;
  bottom?: IndexItem;
  shoes?: IndexItem;
  mono?: IndexItem;
}

function scoreOutfit(
  outfit: Outfit,
  singleScores: Map<string, number>,
  weights: ContextWeights,
): number {
  let score = 0;

  // Sum single item scores
  for (const key of ['top', 'bottom', 'shoes', 'mono'] as const) {
    const it = outfit[key];
    if (it) score += singleScores.get(it.id) ?? 0;
  }

  const top = outfit.top;
  const bottom = outfit.bottom;
  const shoes = outfit.shoes;
  const mono = outfit.mono;

  // Pairwise TN-like factors
  if (top && bottom) {
    const c = pairColorCompatibility(top, bottom);
    const v = pairVibeCompatibility(top, bottom);
    const f = pairFitCompatibility(top, bottom);
    const s = pairSportCompatibility(top, bottom, weights.sportContext);
    const t = pairTeamCompatibility(top, bottom);
    score +=
      WEIGHT_PAIR.color * c +
      WEIGHT_PAIR.vibe * v +
      WEIGHT_PAIR.fit * f +
      WEIGHT_PAIR.sport * s +
      WEIGHT_PAIR.team * t;
  }
  if (top && shoes) {
    const c = pairColorCompatibility(top, shoes);
    const v = pairVibeCompatibility(top, shoes);
    const s = pairSportCompatibility(top, shoes, weights.sportContext);
    const t = pairTeamCompatibility(top, shoes);
    score +=
      WEIGHT_PAIR.color * c +
      WEIGHT_PAIR.vibe * v +
      WEIGHT_PAIR.sport * s +
      WEIGHT_PAIR.team * t;
  }
  if (bottom && shoes) {
    const c = pairColorCompatibility(bottom, shoes);
    const v = pairVibeCompatibility(bottom, shoes);
    const s = pairSportCompatibility(bottom, shoes, weights.sportContext);
    const t = pairTeamCompatibility(bottom, shoes);
    score +=
      WEIGHT_PAIR.color * c +
      WEIGHT_PAIR.vibe * v +
      WEIGHT_PAIR.sport * s +
      WEIGHT_PAIR.team * t;
  }
  if (mono && shoes) {
    const c = pairColorCompatibility(mono, shoes);
    const v = pairVibeCompatibility(mono, shoes);
    const s = pairSportCompatibility(mono, shoes, weights.sportContext);
    const t = pairTeamCompatibility(mono, shoes);
    score +=
      WEIGHT_PAIR.color * c +
      WEIGHT_PAIR.vibe * v +
      WEIGHT_PAIR.sport * s +
      WEIGHT_PAIR.team * t;
  }

  return score;
}

// ------------------------- Main -------------------------

async function main() {
  const idxPath = path.resolve(argv.index as string);
  if (!fs.existsSync(idxPath)) {
    console.error('index file not found:', idxPath);
    process.exit(1);
  }

  const raw = fs.readFileSync(idxPath, 'utf8');
  const items: IndexItem[] = JSON.parse(raw);

  const userPrompt = argv.prompt as string;
  const genderPref = argv.gender_pref as 'any' | 'men' | 'women';

  // Always compute a heuristic intent once so we can fall back or repair Gemini
  const heuristicIntent = guessIntentHeuristic(userPrompt, genderPref);

  let intent = await getIntentFromGemini(userPrompt, genderPref);
  if (!intent) {
    intent = heuristicIntent;
  }

  // If Gemini forgot categories entirely, copy them from the heuristic intent
  if (!intent.required_categories || intent.required_categories.length === 0) {
    intent.required_categories = heuristicIntent.required_categories;
    if (!intent.requested_form) intent.requested_form = heuristicIntent.requested_form;
    if (!intent.outfit_mode) intent.outfit_mode = heuristicIntent.outfit_mode;
  } else {
    // If Gemini says this is an outfit but only gives a single category,
    // and the heuristic intent detected a fuller outfit (e.g. top+bottom+shoes
    // from the word "outfit" / "fit"), then upgrade to the heuristic form.
    const gemCats = new Set(intent.required_categories);
    const heurCats = new Set(heuristicIntent.required_categories);

    const gemIsOutfit = intent.outfit_mode === 'outfit';
    const heurIsOutfit = heuristicIntent.outfit_mode === 'outfit';

    if (gemIsOutfit && heurIsOutfit && gemCats.size < heurCats.size) {
      intent.required_categories = heuristicIntent.required_categories;
      intent.requested_form = heuristicIntent.requested_form;
    }
  }

  const weights = buildWeights(intent);

  logDebug('INTENT', JSON.stringify(intent, null, 2));
  logDebug('WEIGHTS', weights);

  // Build per-role candidate pools
  const requiredCats = Array.from(new Set(intent.required_categories));
  const perRoleLimit = argv.per_role_limit as number;

  const candidatesByCat: Record<CategoryMain, IndexItem[]> = {
    top: [],
    bottom: [],
    shoes: [],
    mono: [],
  };

  for (const cat of requiredCats) {
    const pool = filterCandidatesForRole(items, cat, intent, weights);
    // Score single items first, then restrict to top-N by single score
    const scored = pool.map((it) => ({
      item: it,
      score: scoreSingleItem(it, weights),
    }));
    scored.sort((a, b) => b.score - a.score);
    const limited = scored.slice(0, perRoleLimit).map((s) => s.item);

    candidatesByCat[cat] = limited;
    logDebug(
      `Candidates for ${cat}: ${limited.length} (from ${pool.length})`,
    );
  }

  const singleScores = new Map<string, number>();
  for (const it of items) {
    singleScores.set(it.id, scoreSingleItem(it, weights));
  }

  const epsilon = clamp01(argv.epsilon as number);
  const jitter = argv.jitter as number;
  const poolSize = argv.pool_size as number;

  const outfits: { outfit: Outfit; score: number }[] = [];

  const isOutfit = intent.outfit_mode === 'outfit' && requiredCats.length > 1;

  if (isOutfit) {
    // assemble outfits from candidate pools
    const tops = candidatesByCat.top;
    const bottoms = candidatesByCat.bottom;
    const shoes = candidatesByCat.shoes;
    const monos = candidatesByCat.mono;

    if (requiredCats.includes('mono')) {
      // mono-only or mono+shoes
      const useShoes = requiredCats.includes('shoes');
      for (const mono of monos) {
        if (useShoes && shoes.length) {
          for (const sh of shoes) {
            const outfit: Outfit = { mono, shoes: sh };
            let s = scoreOutfit(outfit, singleScores, weights);
            s += randUniform(-jitter, jitter);
            outfits.push({ outfit, score: s });
          }
        } else {
          const outfit: Outfit = { mono };
          let s = scoreOutfit(outfit, singleScores, weights);
          s += randUniform(-jitter, jitter);
          outfits.push({ outfit, score: s });
        }
      }
    } else {
      // top/bottom(/shoes)
      const useShoes = requiredCats.includes('shoes');
      for (const t of tops) {
        for (const b of bottoms) {
          if (useShoes && shoes.length) {
            for (const sh of shoes) {
              const outfit: Outfit = { top: t, bottom: b, shoes: sh };
              let s = scoreOutfit(outfit, singleScores, weights);
              s += randUniform(-jitter, jitter);
              outfits.push({ outfit, score: s });
            }
          } else {
            const outfit: Outfit = { top: t, bottom: b };
            let s = scoreOutfit(outfit, singleScores, weights);
            s += randUniform(-jitter, jitter);
            outfits.push({ outfit, score: s });
          }
        }
      }
    }
  } else {
    // Single item / partial form: score per requested category independently
    for (const cat of requiredCats) {
      const pool = candidatesByCat[cat];
      for (const it of pool) {
        let s = singleScores.get(it.id) ?? 0;
        s += randUniform(-jitter, jitter);
        const outfit: Outfit = { [cat]: it };
        outfits.push({ outfit, score: s });
      }
    }
  }

  if (!outfits.length) {
    console.error('No outfits/items could be constructed.');
    process.exit(1);
  }

  // Ranking + diversity sampling
  outfits.sort((a, b) => b.score - a.score);
  const topBand = outfits.slice(0, Math.max(poolSize * 3, poolSize));

  const chosen: { outfit: Outfit; score: number }[] = [];
  const usedIds = new Set<string>();

  while (chosen.length < poolSize && topBand.length) {
    let pick: { outfit: Outfit; score: number } | null = null;
    if (Math.random() < epsilon) {
      // pick random from band with weights
      const minScore = topBand[topBand.length - 1].score;
      const scores = topBand.map((o) =>
        Math.max(0.0001, o.score - minScore + 0.001),
      );
      const selected = choiceWeighted(topBand, scores);
      pick = selected || topBand[0];
    } else {
      pick = topBand[0];
    }
    if (!pick) break;

    const ids: string[] = [];
    for (const key of ['top', 'bottom', 'shoes', 'mono'] as const) {
      const it = pick.outfit[key];
      if (it) ids.push(it.id);
    }

    // Avoid duplicate outfits with same items
    const sig = ids.sort().join('|');
    if (!usedIds.has(sig)) {
      usedIds.add(sig);
      chosen.push(pick);
    }

    // Remove that outfit from band
    const idx = topBand.indexOf(pick);
    if (idx >= 0) topBand.splice(idx, 1);
  }

  // Output
  for (let i = 0; i < chosen.length; i++) {
    const { outfit } = chosen[i];
    const order: CategoryMain[] = ['top', 'bottom', 'shoes', 'mono'];
    for (const cat of order) {
      const it = (outfit as any)[cat] as IndexItem | undefined;
      if (it) {
        console.log(`${cat} ${it.imagePath}`);
      }
    }
    if (i !== chosen.length - 1) console.log('');
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});