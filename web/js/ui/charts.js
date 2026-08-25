/**
 * Small inline SVG charts for the diagnostics panel.
 *
 * SVG rather than canvas here: these are static, need to be crisp at any DPI,
 * inherit theme colours through `currentColor` and CSS variables, and are small
 * enough that DOM overhead is irrelevant.
 */

import { escapeHTML } from './mapview.js';

const svg = (w, h, body, extra = '') =>
  `<svg class="chart" viewBox="0 0 ${w} ${h}" width="100%" height="${h}"
        preserveAspectRatio="none" role="img" ${extra}>${body}</svg>`;

/** Grouped horizontal bars comparing a metric before and after correction. */
export function beforeAfterBars(rows) {
  const W = 300, rowH = 30, pad = 4;
  const H = rows.length * rowH + pad * 2;
  const max = Math.max(...rows.flatMap((r) => [Math.abs(r.before), Math.abs(r.after)]), 0.001);
  const body = rows.map((r, i) => {
    const y = pad + i * rowH;
    const bw = (Math.abs(r.before) / max) * (W - 120);
    const aw = (Math.abs(r.after) / max) * (W - 120);
    return `
      <text x="0" y="${y + 10}" font-size="10" fill="var(--fg-subtle)">${escapeHTML(r.label)}</text>
      <rect x="0" y="${y + 14}" width="${bw}" height="4" rx="2" fill="var(--fg-subtle)" opacity=".5"/>
      <rect x="0" y="${y + 20}" width="${aw}" height="4" rx="2" fill="var(--accent)"/>
      <text x="${W - 2}" y="${y + 10}" font-size="10" text-anchor="end"
            fill="var(--fg-muted)" font-family="var(--font-mono)">${r.before.toFixed(3)} → ${r.after.toFixed(3)}</text>`;
  }).join('');
  return svg(W, H, body, 'preserveAspectRatio="xMinYMin meet"');
}

/** Distribution of pairwise cosine distance, with an optional marker. */
export function distributionChart(percentiles, marker) {
  const W = 300, H = 78, padL = 4, padR = 4, padB = 16;
  const pts = Object.entries(percentiles)
    .map(([q, sim]) => ({ q: parseFloat(q), d: 1 - sim }))
    .sort((a, b) => a.d - b.d);
  if (!pts.length) return '<p class="subtle">No distribution recorded in this build.</p>';

  const dMin = pts[0].d, dMax = pts[pts.length - 1].d;
  const x = (d) => padL + ((d - dMin) / (dMax - dMin || 1)) * (W - padL - padR);
  const y = (q) => (H - padB) - (q / 100) * (H - padB - 6);

  const line = pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.d).toFixed(1)},${y(100 - p.q).toFixed(1)}`).join('');
  const area = `${line}L${x(dMax)},${H - padB}L${x(dMin)},${H - padB}Z`;

  let markup = '';
  if (marker != null && isFinite(marker)) {
    const mx = Math.min(Math.max(x(marker), padL), W - padR);
    markup = `<line x1="${mx}" y1="2" x2="${mx}" y2="${H - padB}" stroke="var(--warning)"
                    stroke-width="1.5" stroke-dasharray="3 2"/>
              <text x="${mx}" y="${H - 4}" font-size="9" text-anchor="middle"
                    fill="var(--warning)" font-family="var(--font-mono)">this pair</text>`;
  }
  return svg(W, H, `
    <path d="${area}" fill="var(--accent)" opacity=".14"/>
    <path d="${line}" fill="none" stroke="var(--accent)" stroke-width="1.5"/>
    <text x="${padL}" y="${H - 4}" font-size="9" fill="var(--fg-subtle)"
          font-family="var(--font-mono)">${dMin.toFixed(2)}</text>
    <text x="${W - padR}" y="${H - 4}" font-size="9" text-anchor="end" fill="var(--fg-subtle)"
          font-family="var(--font-mono)">${dMax.toFixed(2)}</text>
    ${markup}`, 'preserveAspectRatio="none"');
}

/** Layer sweep curve, when data/layer_sweep.json is present alongside the build. */
export function sweepChart(rows, chosenLayer) {
  if (!rows || !rows.length) return '';
  const W = 300, H = 110, padL = 22, padR = 6, padT = 8, padB = 18;
  const series = [
    { key: 'probe_accuracy', colour: 'var(--accent)', label: 'probe' },
    { key: 'triplet_accuracy', colour: 'var(--info)', label: 'triplet' },
    { key: 'domain_purity', colour: 'var(--fg-subtle)', label: 'purity' },
  ].filter((s) => rows.some((r) => typeof r[s.key] === 'number'));

  const layers = rows.map((r) => r.layer);
  const lo = Math.min(...layers), hi = Math.max(...layers);
  const x = (l) => padL + ((l - lo) / (hi - lo || 1)) * (W - padL - padR);
  const y = (v) => padT + (1 - v) * (H - padT - padB);

  const paths = series.map((s) => {
    const d = rows.filter((r) => typeof r[s.key] === 'number')
      .map((r, i) => `${i ? 'L' : 'M'}${x(r.layer).toFixed(1)},${y(r[s.key]).toFixed(1)}`).join('');
    return `<path d="${d}" fill="none" stroke="${s.colour}" stroke-width="1.6"
                  stroke-linejoin="round" opacity=".9"/>`;
  }).join('');

  const chosen = chosenLayer != null
    ? `<line x1="${x(chosenLayer)}" y1="${padT}" x2="${x(chosenLayer)}" y2="${H - padB}"
             stroke="var(--warning)" stroke-width="1" stroke-dasharray="3 2"/>`
    : '';

  const legend = series.map((s, i) =>
    `<circle cx="${padL + i * 62}" cy="${H - 5}" r="3" fill="${s.colour}"/>
     <text x="${padL + i * 62 + 7}" y="${H - 2}" font-size="9" fill="var(--fg-subtle)">${s.label}</text>`
  ).join('');

  return svg(W, H, `
    <line x1="${padL}" y1="${y(0)}" x2="${W - padR}" y2="${y(0)}" stroke="var(--border)"/>
    <line x1="${padL}" y1="${y(1)}" x2="${W - padR}" y2="${y(1)}" stroke="var(--border)" stroke-dasharray="2 3"/>
    <text x="2" y="${y(1) + 3}" font-size="9" fill="var(--fg-subtle)" font-family="var(--font-mono)">1.0</text>
    <text x="2" y="${y(0) + 3}" font-size="9" fill="var(--fg-subtle)" font-family="var(--font-mono)">0.0</text>
    ${chosen}${paths}${legend}`, 'preserveAspectRatio="none"');
}
