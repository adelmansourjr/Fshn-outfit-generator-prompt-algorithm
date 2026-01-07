export function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function rmDiacritics(s: string): string {
  return s.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
}

export function normLoose(value: string): string {
  return rmDiacritics(String(value || '').toLowerCase())
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function acronym(value: string): string {
  const STOP = new Set(['fc','cf','club','the','of','and','de','ac','sc','bc','sa','afc','cfm']);
  const words = normLoose(value).split(' ').filter(Boolean);
  if (words.length < 2) return '';
  const filtered = words.filter((w) => !STOP.has(w));
  const pick = filtered.length ? filtered : words;
  return pick.map((w) => w[0]).join('');
}

export function aliases(raw: string): string[] {
  const base = normLoose(raw);
  if (!base) return [];
  const tokens = base.split(' ').filter(Boolean);
  const acr = acronym(base);
  const out = new Set<string>([base, ...tokens]);
  if (acr.length >= 2) out.add(acr);
  if (tokens.length >= 2) out.add(tokens.join(''));
  return Array.from(out);
}

export function uniq<T>(values: T[]): T[] {
  return Array.from(new Set(values));
}

export function hasWord(haystack: string, needle: string): boolean {
  return new RegExp(`\\b${escapeRegExp(needle)}\\b`, 'i').test(haystack);
}

export function textHasKeywordLoose(haystack: string, keyword: string): boolean {
  if (!haystack || !keyword) return false;
  if (hasWord(haystack, keyword)) return true;
  return haystack.includes(keyword);
}
