#!/usr/bin/env ts-node

import fs from 'fs';
import path from 'path';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import vision, { protos } from '@google-cloud/vision';

/** Four-category taxonomy + gender */
type CategoryMain = 'top'|'bottom'|'shoes'|'mono';
type Gender = 'men'|'women'|'unisex';

/** New: silhouette fit for tops & bottoms only */
type Fit = 'oversized'|'regular'|'slim'|'cropped';

/** New: simple sport taxonomy */
type SportType = 'football'|'basketball'|'running'|'tennis'|'gym'|'other';

/** New: sport meta attached only when the item is actually sporty */
type SportMeta = {
  sport: SportType;
  teams: string[];   // team aliases (e.g. ["fc barcelona","barca","fcb"])
  isKit: boolean;    // true for things that look like official kits (home/away/third jersey/shorts)
};

const ALLOWED = {
  colours: ['black','white','grey','red','blue','green','beige','brown','pink','yellow','purple'] as const,
  vibes:   ['streetwear','edgy','minimal','y2k','techwear','sporty','preppy','vintage','chic'] as const,
};
type Colour = typeof ALLOWED.colours[number];
type Vibe   = typeof ALLOWED.vibes[number];

const argv = yargs(hideBin(process.argv))
  .option('images_dir',  { type: 'string', demandOption: true })
  .option('out',         { type: 'string', default: 'index.json' })
  .option('min_colours', { type: 'number', default: 1, describe: 'Guarantee at least this many colours (1–2)' })
  .option('names_json',  { type: 'string', describe: 'Optional JSON mapping { "<filename or base>": "Product Name" }' })
  .parseSync();

type AnnotateImageResponse = protos.google.cloud.vision.v1.IAnnotateImageResponse;
const client = new vision.ImageAnnotatorClient();

/* ===================== Colours ===================== */

const CANON: Record<Colour,[number,number,number]> = {
  black:[20,20,20], white:[235,235,235], grey:[128,128,128], red:[200,30,30],
  blue:[40,80,200], green:[40,160,80], beige:[220,205,180], brown:[120,80,40],
  pink:[230,150,190], yellow:[230,210,40], purple:[140,80,180],
};
function nearestCanon([r,g,b]: [number,number,number]): Colour {
  let best: Colour = 'black', bestD = Infinity;
  for (const [name, c] of Object.entries(CANON) as [Colour,[number,number,number]][]) {
    const d = (c[0]-r)**2 + (c[1]-g)**2 + (c[2]-b)**2;
    if (d < bestD){ bestD = d; best = name; }
  }
  return best;
}

/* ===================== Normalization & Name helpers ===================== */

function rmDiacritics(s: string) {
  return s.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
}
function normalizeForTokens(s: string): string {
  let t = rmDiacritics(s.toLowerCase().trim());
  t = t.replace(/[-_]/g, ' ');          // t-shirt → t shirt
  t = t.replace(/\bone\s*-\s*piece\b/g, 'onepiece');
  t = t.replace(/\bt\s*shirt\b/g, 'tshirt');
  t = t.replace(/\btee\b/g, 'tshirt');
  t = t.replace(/\btrainers?\b/g, 'sneaker');
  t = t.replace(/\bheels?\b/g, 'heel');
  t = t.replace(/\btrouser(s)?\b/g, 'trousers');
  t = t.replace(/\bpant(s)?\b/g, 'pants');
  t = t.replace(/\bflip\s*flops?\b/g, 'slides');
  t = t.replace(/\bgore\s*tex\b/g, 'gore-tex');
  return t;
}

function normLoose(s: string) {
  return normalizeForTokens(s).replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g,' ').trim();
}
function acronym(s: string) {
  const STOP = new Set(['fc','cf','club','the','of','and','de','ac','sc','bc','sa','afc','cfm']);
  const w = normLoose(s).split(' ').filter(Boolean);
  if (w.length < 2) return '';
  const keep = w.filter(x => !STOP.has(x));
  const pick = (keep.length ? keep : w);
  return pick.map(x => x[0]).join('');
}
function aliases(raw: string): string[] {
  const base = normLoose(raw);
  if (!base) return [];
  const toks = base.split(' ').filter(Boolean);
  const acr  = acronym(base);
  const out = new Set<string>([base, ...toks]);
  if (acr.length >= 2) out.add(acr);
  if (toks.length >= 2) out.add(toks.join(''));     // real madrid → realmadrid
  // Barça → barca already via rmDiacritics; keep short populars
  if (base.includes('barcelona')) out.add('barca');
  if (base.includes('real madrid')) out.add('rmcf');
  if (base.includes('manchester united')) out.add('manutd');
  if (base.includes('arsenal')) out.add('afc');
  return Array.from(out);
}

/** Split into unigram + bigram tokens from arbitrary text parts. */
function tokensFromParts(parts: string[]): string[] {
  const normalized = parts.filter(Boolean).map(normalizeForTokens);
  const split = normalized.flatMap(p => p.split(/[^a-z0-9]+/g)).filter(Boolean);
  const bigrams: string[] = [];
  for (const p of normalized) {
    const words = p.split(/[^a-z0-9]+/g).filter(Boolean);
    for (let i = 0; i < words.length - 1; i++) bigrams.push(`${words[i]} ${words[i+1]}`);
  }
  return Array.from(new Set([...normalized, ...split, ...bigrams]));
}

/* ===================== Product name ingestion ===================== */

type NamesMap = Record<string,string>;
function loadNamesMap(p?: string): NamesMap {
  if (!p) return {};
  try {
    const raw = fs.readFileSync(p, 'utf8');
    const obj = JSON.parse(raw);
    if (obj && typeof obj === 'object') return obj as NamesMap;
  } catch {}
  return {};
}
function deriveNameFromFilename(fileBase: string): string {
  // Examples:
  //   "nike_barca_jersey_2023" → "nike barca jersey 2023"
  //   "barcelona-home-kit"     → "barcelona home kit"
  //   "IMG_1234 blue tee"      → "blue tee"
  let n = fileBase;
  n = n.replace(/[_-]+/g, ' ');
  // Drop camera prefixes if they exist
  n = n.replace(/\b(img|dsc|photo|picture)[-_]?\d+\b/gi, '').trim();
  return n || fileBase;
}

/* ===================== Entity extraction (logos + OCR + webEntities + names) ===================== */

type EntityMeta = { text: string; weight: number; type: 'brand'|'team'|'sponsor'|'generic' };
function tagEntityType(e: string): EntityMeta['type'] {
  const t = e;
  // light heuristics, not a hardcoded list dependency
  if (/\b(nike|adidas|puma|balenciaga|timberland|zara|gucci|prada|dior|lv|north face|patagonia|reebok|new balance|h&m|uniqlo)\b/.test(t)) return 'brand';
  if (/\b(fc|cf|real madrid|madrid|barcelona|barca|fcb|arsenal|chelsea|liverpool|psg|bayern|milan|juventus|city|united|tottenham)\b/.test(t)) return 'team';
  if (/\b(emirates|rakuten|spotify|jeep|etihad|qatar|aia|three|fly|beko|yokohama)\b/.test(t)) return 'sponsor';
  return 'generic';
}
function buildEntityBag(srcs: Array<{text:string, score?:number}> | string[]) {
  const list = Array.isArray(srcs) ? (srcs as any[]).map(x => typeof x === 'string' ? {text:x, score:1} : x) : [];
  const out = new Map<string, number>();
  for (const {text, score=1} of list) {
    for (const a of aliases(text)) {
      out.set(a, Math.max(out.get(a) ?? 0, score));
    }
  }
  return Array.from(out.entries()).map(([text,weight]) => ({ text, weight }));
}

/* ===================== Lexicon & Label hints (category) ===================== */

type Hit = { cat: CategoryMain, sub: string, w: number };

const LEXICON = new Map<string, Hit>([
  // MONO
  ['shirt dress', { cat:'mono', sub:'shirt dress',  w: 4 }],
  ['slip dress',  { cat:'mono', sub:'slip dress',   w: 4 }],
  ['wrap dress',  { cat:'mono', sub:'wrap dress',   w: 4 }],
  ['bodycon',     { cat:'mono', sub:'bodycon dress',w: 3 }],
  ['sheath',      { cat:'mono', sub:'bodycon dress',w: 3 }],
  ['a line',      { cat:'mono', sub:'a-line dress', w: 3 }],
  ['gown',        { cat:'mono', sub:'gown',         w: 4 }],
  ['maxi',        { cat:'mono', sub:'maxi dress',   w: 2 }],
  ['midi',        { cat:'mono', sub:'midi dress',   w: 2 }],
  ['mini',        { cat:'mono', sub:'mini dress',   w: 2 }],
  ['jumpsuit',    { cat:'mono', sub:'jumpsuit',     w: 4 }],
  ['romper',      { cat:'mono', sub:'romper',       w: 4 }],
  ['playsuit',    { cat:'mono', sub:'romper',       w: 4 }],
  ['sundress',    { cat:'mono', sub:'sundress',     w: 4 }],
  ['cocktail dress', { cat:'mono', sub:'cocktail dress', w:4 }],
  ['evening dress',  { cat:'mono', sub:'evening dress',  w:4 }],
  ['onepiece',    { cat:'mono', sub:'one-piece',    w: 3 }],
  ['dress',       { cat:'mono', sub:'dress',        w: 3 }],

  // SHOES
  ['sneaker',         { cat:'shoes', sub:'sneakers',    w: 3 }],
  ['running shoe',    { cat:'shoes', sub:'sneakers',    w: 3 }],
  ['basketball shoe', { cat:'shoes', sub:'sneakers',    w: 3 }],
  ['tennis shoe',     { cat:'shoes', sub:'sneakers',    w: 3 }],
  ['boot',            { cat:'shoes', sub:'boots',       w: 3 }],
  ['chelsea',         { cat:'shoes', sub:'boots',       w: 3 }],
  ['combat',          { cat:'shoes', sub:'boots',       w: 3 }],
  ['cowboy boot',     { cat:'shoes', sub:'boots',       w: 3 }],
  ['heel',            { cat:'shoes', sub:'heel',        w: 3 }],
  ['loafers',         { cat:'shoes', sub:'loafers',     w: 3 }],
  ['loafer',          { cat:'shoes', sub:'loafers',     w: 3 }],
  ['oxford',          { cat:'shoes', sub:'oxford/derby',w: 3 }],
  ['derby',           { cat:'shoes', sub:'oxford/derby',w: 3 }],
  ['brogue',          { cat:'shoes', sub:'oxford/derby',w: 3 }],
  ['sandals',         { cat:'shoes', sub:'sandals/slides', w:3 }],
  ['slides',          { cat:'shoes', sub:'sandals/slides', w:3 }],
  ['mules',           { cat:'shoes', sub:'sandals/slides', w:3 }],
  ['mary jane',       { cat:'shoes', sub:'mary jane',   w: 3 }],
  ['shoe',            { cat:'shoes', sub:'shoes',       w: 1 }],

  // BOTTOM — incl. leather pants/leggings
  ['leather trousers', { cat:'bottom', sub:'leather trousers', w: 5 }],
  ['leather pants',    { cat:'bottom', sub:'leather trousers', w: 5 }],
  ['leather leggings', { cat:'bottom', sub:'leather leggings', w: 4 }],
  ['mom jeans',        { cat:'bottom', sub:'mom jeans',        w: 4 }],
  ['bootcut',          { cat:'bottom', sub:'bootcut jeans',    w: 3 }],
  ['wide leg',         { cat:'bottom', sub:'wide-leg jeans',   w: 3 }],
  ['baggy',            { cat:'bottom', sub:'wide-leg jeans',   w: 3 }],
  ['loose fit',        { cat:'bottom', sub:'wide-leg jeans',   w: 3 }],
  ['straight leg',     { cat:'bottom', sub:'straight-leg jeans',w:3 }],
  ['skinny',           { cat:'bottom', sub:'skinny jeans',     w: 3 }],
  ['jeans',            { cat:'bottom', sub:'jeans',            w: 3 }],
  ['denim',            { cat:'bottom', sub:'jeans',            w: 2 }],
  ['trousers',         { cat:'bottom', sub:'trousers',         w: 3 }],
  ['pants',            { cat:'bottom', sub:'trousers',         w: 3 }],
  ['chinos',           { cat:'bottom', sub:'trousers',         w: 3 }],
  ['slacks',           { cat:'bottom', sub:'trousers',         w: 3 }],
  ['cargo',            { cat:'bottom', sub:'cargo trousers',   w: 3 }],
  ['joggers',          { cat:'bottom', sub:'joggers',          w: 3 }],
  ['sweatpants',       { cat:'bottom', sub:'joggers',          w: 3 }],
  ['track pants',      { cat:'bottom', sub:'track pants',      w: 2 }],
  ['leggings',         { cat:'bottom', sub:'leggings',         w: 2 }],
  ['pleated skirt',    { cat:'bottom', sub:'pleated skirt',    w: 3 }],
  ['denim skirt',      { cat:'bottom', sub:'denim skirt',      w: 3 }],
  ['mini skirt',       { cat:'bottom', sub:'mini skirt',       w: 2 }],
  ['midi skirt',       { cat:'bottom', sub:'midi skirt',       w: 2 }],
  ['maxi skirt',       { cat:'bottom', sub:'maxi skirt',       w: 2 }],
  ['skirt',            { cat:'bottom', sub:'skirt',            w: 2 }],
  ['shorts',           { cat:'bottom', sub:'shorts',           w: 3 }],
  ['bermuda shorts',   { cat:'bottom', sub:'shorts',           w: 3 }],
  ['culottes',         { cat:'bottom', sub:'culottes/skort',   w: 2 }],
  ['skort',            { cat:'bottom', sub:'culottes/skort',   w: 2 }],
  ['bottom',           { cat:'bottom', sub:'bottom',           w: 1 }],

  // TOP
  ['puffer jacket', { cat:'top', sub:'puffer jacket', w: 3 }],
  ['down jacket',   { cat:'top', sub:'puffer jacket', w: 3 }],
  ['quilted jacket',{ cat:'top', sub:'puffer jacket', w: 3 }],
  ['trench coat',   { cat:'top', sub:'trench coat',   w: 3 }],
  ['parka',         { cat:'top', sub:'parka',         w: 3 }],
  ['denim jacket',  { cat:'top', sub:'denim jacket',  w: 3 }],
  ['leather jacket',{ cat:'top', sub:'leather jacket',w: 4 }],
  ['leather blazer',{ cat:'top', sub:'leather blazer',w: 3 }],
  ['leather coat',  { cat:'top', sub:'leather coat',  w: 3 }],
  ['bomber jacket', { cat:'top', sub:'bomber jacket', w: 3 }],
  ['coat',          { cat:'top', sub:'coat',          w: 2 }],
  ['blazer',        { cat:'top', sub:'blazer',        w: 2 }],
  ['gilet',         { cat:'top', sub:'gilet/vest',    w: 2 }],
  ['vest',          { cat:'top', sub:'gilet/vest',    w: 2 }],
  ['cardigan',      { cat:'top', sub:'cardigan',      w: 3 }],
  ['hoodie',        { cat:'top', sub:'hoodie',        w: 2 }],
  ['sweatshirt',    { cat:'top', sub:'sweatshirt',    w: 3 }],
  ['crewneck',      { cat:'top', sub:'sweatshirt',    w: 3 }],
  ['fleece',        { cat:'top', sub:'sweatshirt',    w: 3 }],
  ['sweater',       { cat:'top', sub:'sweater',       w: 3 }],
  ['jumper',        { cat:'top', sub:'sweater',       w: 3 }],
  ['knitwear',      { cat:'top', sub:'sweater',       w: 3 }],
  ['pullover',      { cat:'top', sub:'sweater',       w: 3 }],
  ['oxford shirt',  { cat:'top', sub:'shirt',         w: 3 }],
  ['dress shirt',   { cat:'top', sub:'shirt',         w: 3 }],
  ['button down',   { cat:'top', sub:'shirt',         w: 3 }],
  ['shirt',         { cat:'top', sub:'shirt/blouse',  w: 2 }],
  ['blouse',        { cat:'top', sub:'shirt/blouse',  w: 2 }],
  ['polo',          { cat:'top', sub:'polo',          w: 2 }],
  ['rugby shirt',   { cat:'top', sub:'rugby shirt',   w: 2 }],
  ['tshirt',        { cat:'top', sub:'t-shirt',       w: 3 }],
  ['graphic tee',   { cat:'top', sub:'t-shirt',       w: 3 }],
  ['tank',          { cat:'top', sub:'tank/cami',     w: 2 }],
  ['camisole',      { cat:'top', sub:'tank/cami',     w: 2 }],
  ['cami',          { cat:'top', sub:'tank/cami',     w: 2 }],
  ['tube top',      { cat:'top', sub:'tank/cami',     w: 2 }],
  ['crop top',      { cat:'top', sub:'tank/cami',     w: 2 }],
  ['jersey',        { cat:'top', sub:'jersey',        w: 2 }],
  ['sports bra',    { cat:'top', sub:'sports bra',    w: 2 }],
  ['top',           { cat:'top', sub:'top',           w: 1 }],
  // Accessories (fold to top)
  ['handbag', { cat:'top', sub:'accessory', w: 1 }],
  ['bag',     { cat:'top', sub:'accessory', w: 1 }],
  ['cap',     { cat:'top', sub:'accessory', w: 1 }],
  ['beanie',  { cat:'top', sub:'accessory', w: 1 }],
  ['belt',    { cat:'top', sub:'accessory', w: 1 }],
  ['scarf',   { cat:'top', sub:'accessory', w: 1 }],
  ['glove',   { cat:'top', sub:'accessory', w: 1 }],
]);

const LABEL_HINTS: Array<[RegExp, Hit]> = [
  [/dress(?!\s+shoe)/i,          { cat:'mono',   sub:'dress',              w: 5 }],
  [/jumpsuit|romper|playsuit/i,  { cat:'mono',   sub:'jumpsuit/romper',    w: 5 }],
  [/skirt/i,                     { cat:'bottom', sub:'skirt',              w: 4 }],
  [/shorts?/i,                   { cat:'bottom', sub:'shorts',             w: 4 }],
  [/\b(jeans|denim)\b/i,         { cat:'bottom', sub:'jeans',              w: 5 } as any],
  [/trousers?|pants|slacks/i,    { cat:'bottom', sub:'trousers',           w: 5 }],
  [/leggings/i,                  { cat:'bottom', sub:'leggings',           w: 4 }],
  [/footwear|shoe|sneaker|boot|heel|loafer|oxford|derby|sandal|slide/i,
                                  { cat:'shoes',  sub:'shoes',              w: 5 }],
  [/outerwear|jacket|coat|parka|trench|blazer|gilet|vest/i,
                                  { cat:'top',    sub:'outerwear',          w: 4 }],
  [/shirt|blouse|polo|t[- ]?shirt|tee/i,
                                  { cat:'top',    sub:'shirt/t-shirt',      w: 4 }],
  [/sweater|jumper|cardigan|sweatshirt|fleece|crewneck|pullover/i,
                                  { cat:'top',    sub:'knit/sweat',         w: 4 }],
];

/* ===================== Category decision ===================== */

type CatSub = { category: CategoryMain, sub: string };

function decideCategory(labels: string[], tokens: string[]): CatSub {
  const scores: Record<CategoryMain, Record<string, number>> = { top:{}, bottom:{}, shoes:{}, mono:{} };
  const bump = (c: CategoryMain, s: string, w: number) => { scores[c][s] = (scores[c][s] ?? 0) + w; };

  const tJoined = ` ${tokens.join(' ')} `;

  // Guard: "dress shoes" → shoes, never mono
  if (/\bdress\s+(shoe|shoes|oxford|derby|loafer|heel|heels|pump|pumps|boot|boots|sandal|sandals|slide|slides)\b/i.test(tJoined)) {
    bump('shoes','formal shoes', 10);
  }

  // Label hints
  for (const desc of labels) {
    for (const [re, hit] of LABEL_HINTS) {
      if (re.test(desc)) bump(hit.cat, hit.sub, hit.w);
    }
  }

  // Lexicon hits
  for (const tok of tokens) {
    const hit = LEXICON.get(tok);
    if (hit) bump(hit.cat, hit.sub, hit.w);
  }

  // Context tweaks
  const hoodEvidence = /\b(hood|hooded|drawstring|kangaroo|zip|zipper)\b/i.test(tJoined);
  if (scores.top['hoodie']) scores.top['hoodie'] *= hoodEvidence ? 1.6 : 0.5;

  const leatherBottom = /\b(leather|pleather|faux\s*leather)\b/i.test(tJoined)
                     && /\b(pants?|trousers?|leggings)\b/i.test(tJoined);
  if (leatherBottom) {
    bump('bottom','leather trousers', 3);
    if (scores.top['leather jacket']) scores.top['leather jacket'] *= 0.7;
    if (scores.top['leather blazer']) scores.top['leather blazer'] *= 0.7;
    if (scores.top['leather coat'])   scores.top['leather coat']   *= 0.7;
  }

  const hasMonoWord = /\b(dress|gown|jumpsuit|onepiece|romper|playsuit)\b/i.test(tJoined)
                   && !/\bdress\s+(shoe|shoes|oxford|derby|loafer|heel|heels|pump|pumps|boot|boots|sandal|sandals|slide|slides)\b/i.test(tJoined);
  if (hasMonoWord) {
    for (const k of Object.keys(scores.top)) scores.top[k] *= 0.6;
    bump('mono','dress', 1.5);
  }

  // Mutually exclusive dampers
  const anyTopHit    = Object.values(scores.top).some(v => v > 0);
  const anyBottomHit = Object.values(scores.bottom).some(v => v > 0);
  const anyShoeHit   = Object.values(scores.shoes).some(v => v > 0);
  const anyMonoHit   = Object.values(scores.mono).some(v => v > 0);

  if (anyBottomHit && !anyTopHit && !anyMonoHit && !anyShoeHit) {
    if (scores.top['shirt']           ) scores.top['shirt']            *= 0.4;
    if (scores.top['shirt/blouse']    ) scores.top['shirt/blouse']     *= 0.4;
    if (scores.top['t-shirt']         ) scores.top['t-shirt']          *= 0.4;
    if (scores.top['tank/cami']       ) scores.top['tank/cami']        *= 0.4;
  }
  if (anyShoeHit && !anyMonoHit) {
    for (const k of Object.keys(scores.mono)) scores.mono[k] *= 0.5;
  }

  const catOrder: CategoryMain[] = ['mono','shoes','bottom','top'];
  const catBest: Record<CategoryMain, { sub: string, score: number }> = {
    top:    { sub:'top',    score: 0 },
    bottom: { sub:'bottom', score: 0 },
    shoes:  { sub:'shoes',  score: 0 },
    mono:   { sub:'dress',  score: 0 },
  };
  for (const cat of Object.keys(scores) as CategoryMain[]) {
    const entries = Object.entries(scores[cat]);
    if (!entries.length) continue;
    entries.sort((a,b)=> b[1]-a[1]);
    catBest[cat] = { sub: entries[0][0], score: entries[0][1] };
  }

  const MIN = 1.25;
  const ranked = (Object.keys(catBest) as CategoryMain[])
    .filter(c => catBest[c].score >= MIN)
    .sort((a,b) => {
      if (catBest[b].score !== catBest[a].score) return catBest[b].score - catBest[a].score;
      return catOrder.indexOf(a) - catOrder.indexOf(b);
    });

  if (ranked.length) return { category: ranked[0], sub: catBest[ranked[0]].sub };

  // Fallbacks
  if (hasMonoWord) return { category:'mono', sub:'dress' };
  if (/\b(shoe|sneaker|boot|heel|loafer|derby|oxford|sandal|slide)s?\b/i.test(tJoined)) {
    return { category:'shoes', sub:'shoes' };
  }
  if (/\b(jeans|denim|trousers|pants|chinos?|cargo|shorts?|skirt|leggings?)\b/i.test(tJoined)) {
    if (/\b(leather|pleather|faux\s*leather)\b/i.test(tJoined)) return { category:'bottom', sub:'leather trousers' };
    return { category:'bottom', sub:'bottom' };
  }
  return { category:'top', sub:'top' };
}

/* ===================== Gender decision ===================== */

function decideGender(labels: string[], tokens: string[], cat: CategoryMain, sub: string): Gender {
  const t = ` ${tokens.join(' ')} `;
  const ljoined = ` ${labels.join(' ').toLowerCase()} `;

  const menWords = [
    'men','mens','men s',"men's",'male','man','guys','boys','menswear','for men','menwear',
    'dress shoe','oxford','derby','brogue'
  ];
  const womenWords = [
    'women','womens','women s',"women's",'female','woman','ladies','lady','girls','womenswear','for women','womenwear',
    'maternity'
  ];
  const unisexWords = ['unisex','all gender','all genders','gender neutral','gender-neutral','any gender'];

  let men = 0, women = 0, uni = 0;

  if (unisexWords.some(w => t.includes(w))) uni += 6;
  if (menWords.some(w => t.includes(w) || ljoined.includes(` ${w} `))) men += 5;
  if (womenWords.some(w => t.includes(w) || ljoined.includes(` ${w} `))) women += 5;

  const womenSubs = new Set([
    'dress','shirt dress','slip dress','wrap dress','bodycon dress','a-line dress','gown',
    'maxi dress','midi dress','mini dress','one-piece','jumpsuit','romper','sundress','cocktail dress','evening dress',
    'skirt','pleated skirt','denim skirt','mini skirt','midi skirt','maxi skirt',
    'sports bra','tank/cami','camisole','cami','tube top','crop top','mary jane','heels','heel',
    'leather leggings','leggings','mom jeans'
  ]);
  if (cat === 'mono') women += 5;
  if (womenSubs.has(sub)) women += 4;

  const menSubs = new Set([
    'oxford/derby','oxford','derby','rugby shirt','dress shirt','oxford shirt'
  ]);
  if (menSubs.has(sub)) men += 3;

  const likelyUnisexSubs = new Set([
    't-shirt','hoodie','sweatshirt','sweater','cardigan','polo','jersey',
    'bomber jacket','denim jacket','puffer jacket','trench coat','parka','coat','blazer','leather jacket','leather blazer','leather coat',
    'jeans','cargo trousers','trousers','track pants','shorts','gilet/vest','accessory'
  ]);
  if (likelyUnisexSubs.has(sub) || cat === 'shoes') uni += 2;

  const tokenHasMen = menWords.some(w => t.includes(w) || ljoined.includes(` ${w} `));
  const tokenHasWomen = womenWords.some(w => t.includes(w) || ljoined.includes(` ${w} `));
  if (tokenHasMen && tokenHasWomen) uni += 4;

  if (uni >= 6) return 'unisex';

  const margin = 2;
  if (women >= men + margin) return 'women';
  if (men >= women + margin) return 'men';

  if (cat === 'mono') return 'women';
  if (sub.includes('skirt') || sub.includes('sports bra')) return 'women';
  if (sub.includes('oxford') || sub.includes('derby')) return 'men';

  return 'unisex';
}

/* ===================== Vibes ===================== */

function vibesFromTokens(tokens: string[], category: CategoryMain, colours: Colour[]): Vibe[] {
  const has = (...keys: string[]) => keys.some(k => tokens.some(t => t.includes(k)));
  const out = new Set<Vibe>();

  if (has('streetwear','street','urban','skate','hypebeast','graphic','logo')) out.add('streetwear');
  if (has('sport','sportswear','athletic','training','running','gym','basketball','soccer','tennis','performance')) out.add('sporty');
  if (has('minimal','minimalist','clean','basic','plain','simple','essentials','capsule')) out.add('minimal');
  if (has('vintage','retro','heritage','classic','oldschool','90s','80s','70s','60s','throwback')) out.add('vintage');
  if (has('y2k','2000s',"2000's",'early 2000')) out.add('y2k');
  if (has('preppy','ivy','collegiate','varsity','prep','argyle','polo')) out.add('preppy');
  if (has('chic','elegant','evening','formal','cocktail','luxury','silk','satin','slip')) out.add('chic');
  if (has('edgy','punk','grunge','goth','rock','metal','biker','ripped','distressed','leather')) out.add('edgy');
  if (has('techwear','gore-tex','shell','waterproof','windproof','tactical','utility','cargo','nylon')) out.add('techwear');

  if (category === 'shoes' && has('sneaker','trainer','running')) { out.add('sporty'); out.add('streetwear'); }
  if (category === 'mono'  && (has('slip','satin','evening','cocktail') || colours.includes('black'))) out.add('chic');
  if (category === 'top'   && has('hoodie','sweatshirt','graphic','jersey')) out.add('streetwear');
  if (category === 'bottom'&& has('cargo','ripped','distressed')) out.add('edgy');

  if (colours.includes('black') && (has('leather','biker','punk','grunge') || category === 'top')) out.add('edgy');
  if ((colours.includes('white') || colours.includes('beige') || colours.includes('grey')) && has('plain','basic','clean','minimal')) out.add('minimal');

  if (out.size === 0) {
    if (category === 'shoes') out.add(has('boot') ? 'edgy' : 'sporty');
    else if (category === 'mono') out.add('chic');
    else if (category === 'top' || category === 'bottom') {
      if (has('hoodie','graphic','logo','cargo','denim')) out.add('streetwear');
      else out.add('minimal');
    }
  }
  const allowed = Array.from(out).filter(v => (ALLOWED.vibes as readonly string[]).includes(v));
  return allowed.slice(0, 2);
}

/* ===================== Colour parsing & extraction ===================== */

function colourSynonymToCanon(word: string): Colour | null {
  const w = word.toLowerCase();
  if ((ALLOWED.colours as readonly string[]).includes(w as Colour)) return w as Colour;
  if (['navy','indigo','azure','cobalt','sky'].some(x => w.includes(x))) return 'blue';
  if (['cream','ecru','ivory','offwhite','off-white','off white','bone','oat','sand','khaki','camel','tan'].some(x => w.includes(x))) return 'beige';
  if (['maroon','burgundy','crimson','scarlet','wine'].some(x => w.includes(x))) return 'red';
  if (['chartreuse','lime','olive','forest','emerald','sage','mint'].some(x => w.includes(x))) return 'green';
  if (['fuchsia','magenta','rose','blush','salmon','coral'].some(x => w.includes(x))) return 'pink';
  if (['gold','mustard','amber','lemon','sunflower'].some(x => w.includes(x))) return 'yellow';
  if (['violet','lilac','lavender','plum','mauve'].some(x => w.includes(x))) return 'purple';
  if (['charcoal','graphite','slate'].some(x => w.includes(x))) return 'grey';
  if (['chocolate','espresso','coffee','walnut','mahogany'].some(x => w.includes(x))) return 'brown';
  if (['offblack','off-black'].some(x => w.includes(x))) return 'black';
  return null;
}

function coloursFromTokens(tokens: string[], limit = 2): Colour[] {
  const out: Colour[] = [];
  for (const t of tokens) {
    const c = colourSynonymToCanon(t);
    if (c && !out.includes(c)) out.push(c);
    if (out.length >= limit) break;
  }
  return out;
}

async function averageCanonViaSharpCenter(filePath: string): Promise<Colour | null> {
  try {
    const sharp = (await import('sharp')).default;
    const img = sharp(filePath);
    const meta = await img.metadata();
    const w = meta.width || 256, h = meta.height || 256;

    const side = Math.floor(Math.min(w, h) * 0.6);
    const left = Math.floor((w - side) / 2);
    const top  = Math.floor((h - side) / 2);

    const { data, info } = await img.extract({ left, top, width: side, height: side })
      .resize(64,64,{ fit: 'cover' })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    let r = 0, g = 0, b = 0;
    for (let i = 0; i < info.width * info.height; i++) {
      r += data[3*i]; g += data[3*i+1]; b += data[3*i+2];
    }
    const n = info.width * info.height || 1;
    return nearestCanon([Math.round(r/n), Math.round(g/n), Math.round(b/n)]);
  } catch { return null; }
}

async function extractColours(
  filePath: string,
  propRes: AnnotateImageResponse,
  tokens: string[],
  minColours: number
): Promise<Colour[]> {
  const dom = (propRes.imagePropertiesAnnotation?.dominantColors?.colors || []).slice();
  const sorted = dom.sort((a,b) => (b.score||0)-(a.score||0)).slice(0,3);
  const colours: Colour[] = [];
  for (const c of sorted) {
    const rgb: [number,number,number] = [
      Math.round(c.color?.red ?? 0),
      Math.round(c.color?.green ?? 0),
      Math.round(c.color?.blue ?? 0),
    ];
    const name = nearestCanon(rgb);
    if (!colours.includes(name)) colours.push(name);
    if (colours.length === 2) break;
  }
  if (colours.length === 0) {
    const byTokens = coloursFromTokens(tokens, 2);
    for (const t of byTokens) if (!colours.includes(t)) colours.push(t);
  }
  if (colours.length < Math.max(1, Math.min(2, minColours))) {
    const avg = await averageCanonViaSharpCenter(filePath);
    if (avg && !colours.includes(avg)) colours.push(avg);
  }
  if (colours.length === 0) colours.push('grey');
  return colours.slice(0, 2);
}

/* ===================== New helpers: FIT + SPORT META ===================== */

/** Infer silhouette fit for tops & bottoms only */
function inferFit(category: CategoryMain, sub: string, tokens: string[]): Fit | null {
  if (category !== 'top' && category !== 'bottom') return null;

  const joined = ` ${tokens.join(' ')} `;

  // Cropped first (strong semantic)
  if (/\b(crop|cropped)\b/.test(joined) || sub.includes('crop top')) {
    return 'cropped';
  }

  // Oversized / relaxed / baggy / wide
  if (/\b(oversized|over sized|boxy|slouchy|relaxed|loose fit|loose|baggy|wide leg|wide-leg)\b/.test(joined)) {
    return 'oversized';
  }
  if (sub.includes('wide-leg') || sub.includes('wide leg') || sub.includes('baggy')) {
    return 'oversized';
  }

  // Slim / skinny / fitted
  if (/\b(skinny|slim fit|slim-fit|slim|fitted|tapered)\b/.test(joined) || sub.includes('skinny')) {
    return 'slim';
  }

  // Default if we have no strong evidence
  return 'regular';
}

/** Infer main sport type from tokens + labels */
function inferSportType(tokens: string[], labels: string[]): SportType | null {
  const joined = ` ${tokens.join(' ')} ${labels.join(' ')} `;

  if (/\b(football|soccer|futbol|fútbol)\b/.test(joined)) return 'football';
  if (/\b(basketball|nba)\b/.test(joined)) return 'basketball';
  if (/\b(tennis)\b/.test(joined)) return 'tennis';
  if (/\b(running|runner|jogging|marathon|track and field)\b/.test(joined)) return 'running';
  if (/\b(gym|training|workout|fitness|crossfit)\b/.test(joined)) return 'gym';
  if (/\bsport|sportswear|athletic\b/.test(joined)) return 'other';
  return null;
}

/** Decide if item is sporty and, if so, return sport meta (tops/bottoms/shoes). */
function inferSportMeta(
  category: CategoryMain,
  sub: string,
  vibes: Vibe[],
  tokens: string[],
  labels: string[],
  entityMeta: EntityMeta[]
): SportMeta | null {
  const joinedTokens = ` ${tokens.join(' ')} `;
  const joinedLabels = ` ${labels.join(' ')} `;

  const sportFromText = inferSportType(tokens, labels);

  const hasSportWords = /\b(football|soccer|basketball|tennis|running|runner|jogging|athletic|training|gym|sportswear|sport)\b/
    .test(joinedTokens + joinedLabels);

  const isJerseyLike = sub.includes('jersey');
  const isTrackPants = sub.includes('track pants') || sub.includes('joggers') || sub.includes('shorts');
  const isSportBra   = sub.includes('sports bra');

  const vibesSporty = vibes.includes('sporty');

  const isSporty =
    vibesSporty ||
    !!sportFromText ||
    hasSportWords ||
    isJerseyLike ||
    isTrackPants ||
    isSportBra;

  if (!isSporty) return null;

  const sport: SportType = sportFromText || 'other';
  const teams = entityMeta.filter(e => e.type === 'team').map(e => e.text);

  const kitWords = /\b(kit|home|away|third|jersey)\b/.test(joinedTokens + joinedLabels);
  const isKit = (category === 'top' || category === 'bottom') && teams.length > 0 && kitWords;

  return {
    sport,
    teams,
    isKit,
  };
}

/* ===================== Main ===================== */

async function main() {
  const dir = argv.images_dir!;
  if (!fs.existsSync(dir)) { console.error('images_dir not found:', dir); process.exit(1); }

  const minColours = Math.max(1, Math.min(2, argv.min_colours as number));
  const images = fs.readdirSync(dir).filter(f => /\.(jpe?g|png)$/i.test(f));

  // Optional names map
  const namesMap = loadNamesMap(argv.names_json);

  const index: Array<{
    id: string;
    imagePath: string;
    category: CategoryMain;
    sub?: string|null;
    colours: Colour[];
    vibes: Vibe[];
    gender: Gender;
    /** New fields */
    fit?: Fit|null;
    sportMeta?: SportMeta|null;
    /** Existing name/entity metadata */
    name?: string|null;
    name_normalized?: string|null;
    entities?: string[];
    entityMeta?: EntityMeta[];
  }> = [];

  for (const fname of images) {
    const filePath = path.join(dir, fname);
    const fileBase = path.basename(fname, path.extname(fname));

    // Resolve product name: names.json (exact filename or base) → fallback to filename-derived
    const nameFromMap = namesMap[fname] || namesMap[fileBase];
    const resolvedName = nameFromMap || deriveNameFromFilename(fileBase);
    const resolvedNameNorm = normLoose(resolvedName);

    // Vision calls
    const [labelRes] = await client.labelDetection(filePath);
    const [propRes]  = await client.imageProperties(filePath);
    const [webRes]   = await client.webDetection(filePath);
    const [logoRes]  = await client.logoDetection(filePath);
    const [textRes]  = await client.textDetection(filePath);

    // Labels (raw + normalized for tokens)
    const labelDescs = (labelRes.labelAnnotations || []).map(a => (a.description || '').toLowerCase()).filter(Boolean);
    const labelsForTokens = labelDescs.map(normalizeForTokens);
    const labelsForDecision = labelDescs.slice();

    // Web entities & Best guesses
    const webEntities = (webRes.webDetection?.webEntities || []).map(e => e.description || '').filter(Boolean);
    const bestGuess   = (webRes.webDetection?.bestGuessLabels || []).map(b => b.label || '').filter(Boolean);

    // Logos & OCR
    const logos = (logoRes.logoAnnotations || [])
      .map(a => ({ text: a.description || '', score: a.score ?? 0 }))
      .filter(x => x.text);

    const ocrRaw = (textRes.textAnnotations?.[0]?.description || '').trim();

    // ENTITIES (logos + web entities + ocr + fileBase + product name)
    const entityBag = [
      ...buildEntityBag(logos),
      ...buildEntityBag(webEntities.map(text => ({ text, score: 0.9 }))),
      ...buildEntityBag(bestGuess.map(text => ({ text, score: 0.8 }))),
      ...buildEntityBag([{ text: resolvedName, score: 0.9 }]),
      ...buildEntityBag([{ text: fileBase,     score: 0.7 }]),
      ...buildEntityBag([{ text: ocrRaw,       score: 0.7 }]),
    ];
    // Dedup by text, keep max weight
    const entityMap = new Map<string, number>();
    for (const {text, weight} of entityBag) {
      entityMap.set(text, Math.max(entityMap.get(text) ?? 0, weight));
    }
    const entities = Array.from(entityMap.keys());
    const entityMeta: EntityMeta[] = entities.map(text => ({ text, weight: entityMap.get(text) || 1, type: tagEntityType(text) }));

    // TOKENS for classification: labels + web entities + bestGuess + fileBase + product name + OCR
    const tokens = tokensFromParts([
      ...labelsForTokens,
      ...webEntities.map(normalizeForTokens),
      ...bestGuess.map(normalizeForTokens),
      normalizeForTokens(fileBase),
      normalizeForTokens(resolvedName),
      normalizeForTokens(ocrRaw),
    ]);

    // Decide category
    const { category, sub } = decideCategory(labelsForDecision, tokens);
    // Decide gender (tokens already include name & OCR)
    const gender = decideGender(labelsForDecision, tokens, category, sub);

    // Colours & vibes
    const colours = await extractColours(
      filePath,
      { imagePropertiesAnnotation: (propRes as any).imagePropertiesAnnotation } as any,
      tokens,
      minColours
    );
    const vibes = vibesFromTokens(tokens, category, colours);

    // New: silhouette fit (tops & bottoms only)
    const fit = inferFit(category, sub, tokens);

    // New: sport meta (only populated when sporty; works for tops/bottoms/shoes)
    const sportMeta = inferSportMeta(category, sub, vibes, tokens, labelsForDecision, entityMeta);

    index.push({
      id: fname,
      imagePath: filePath,
      category,
      sub: sub ?? null,
      colours,
      vibes,
      gender,
      fit: fit ?? null,
      sportMeta: sportMeta ?? null,
      name: resolvedName || null,
      name_normalized: resolvedNameNorm || null,
      entities,
      entityMeta,
    });

    const entsForLog = entityMeta
      .sort((a,b)=> (b.weight-a.weight))
      .slice(0,5)
      .map(e => `${e.text}${e.type!=='generic' ? `(${e.type})` : ''}`)
      .join(', ');

    const sportLog = sportMeta ? `${sportMeta.sport}${sportMeta.isKit ? ' (kit)' : ''}${sportMeta.teams.length ? ' ['+sportMeta.teams.join('|')+']' : ''}` : '-';

    console.log(
      `Tagged ${fname} → cat=${category}/${sub || '-'} | gender=${gender}` +
      ` | fit=${fit || '-'} | sport=${sportLog}` +
      ` | colours=${colours.join(',')}` +
      ` | vibes=${vibes.join(',') || '-'}` +
      ` | name="${resolvedName}" | entities=[${entsForLog}]`
    );
  }

  fs.writeFileSync(argv.out as string, JSON.stringify(index, null, 2), 'utf8');
  console.log(`\nWrote ${argv.out} with ${index.length} items.`);
}

main().catch(err => { console.error(err); process.exit(1); });