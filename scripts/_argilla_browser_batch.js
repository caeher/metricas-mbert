const list = await browser.pages.list();
const hit = list.find(p => p.url.includes('annotation-mode'));
if (!hit) throw new Error('No annotation tab');
const page = browser.page(hit.pageId);
const I = {
  sa: 'annotation_status_anotado',
  si: 'annotation_status_contexto_insuficiente',
  l0: 'label_0',
  l1: 'label_1',
  l2: 'label_2',
  l3: 'label_3',
  d: 'dialect_salvadoreno',
  du: 'dialect_desconocido',
  qy: 'quoted_or_reported_yes',
  qn: 'quoted_or_reported_no',
  qunk: 'quoted_or_reported_unknown',
  ny: 'negated_yes',
  nn: 'negated_no',
  nunk: 'negated_unknown',
  vn: 'violence_type_none',
  vw: 'violence_type_wish',
  vi: 'violence_type_incitement',
  vt: 'violence_type_direct_threat',
  ve: 'violence_type_endorsement',
  qa: 'quality_usable',
  qi: 'quality_insufficient_context',
  qr: 'quality_review',
  io: 'identity_targets_origen_nacionalidad',
  ig: 'identity_targets_genero_sexualidad',
  ip: 'identity_targets_afiliacion_politica',
  ir: 'identity_targets_raza_etnia',
  ic: 'identity_targets_clase_socioeconomica',
  irel: 'identity_targets_religion',
  id: 'identity_targets_discapacidad',
};
async function getInfo() {
  const md = await page.read();
  const c = md.match(/(\d+)\s+of\s+(\d+)/i);
  const i = md.indexOf('Texto original');
  const text =
    i >= 0
      ? md
          .slice(i + 14)
          .split('Texto normalizado')[0]
          .replace(/\s+/g, ' ')
          .trim()
      : '';
  const empty = /no tienes registros pendientes|no pending/i.test(md);
  return {
    counter: c?.[0],
    n: parseInt(c?.[1], 10),
    total: parseInt(c?.[2], 10),
    text,
    empty,
  };
}
async function ck(id) {
  if (!id) return;
  await page.evaluate(id => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ block: 'center' });
      window.scrollBy(0, 90);
      el.click();
    }
  }, id);
}
async function fill(q, t) {
  await page.evaluate(
    ({ q, t }) => {
      const el = [...document.querySelectorAll('[aria-label^="Question:"]')]
        .find(x => x.getAttribute('aria-label').endsWith(q))
        ?.querySelector('[contenteditable="true"]');
      if (el) {
        el.focus();
        el.textContent = '';
        el.dispatchEvent(new Event('input', { bubbles: true }));
        document.execCommand('insertText', false, t);
        el.dispatchEvent(new Event('input', { bubbles: true }));
      }
    },
    { q, t }
  );
}
function decide(text) {
  const t = text.toLowerCase();
  const out = {};
  const hasSlur = /\b(cerote|cerota|jueputa|vergaso|pendejo|culero|hijo de puta|malparido|come mierda|mapachon|sorete|puta|puto|maric|perra|zorra|mierda)\b/.test(
    t
  );
  if (/\b(te voy a matar|te mato|voy a matarte|vamos a matar|hay que matarlos|matarlos a todos|mandarlos a matar|guillotina|linch|ejecuten|quemarles la casa)\b/.test(t))
    return { ...out, a: '3', v: 'vi', r: 'Amenaza o incitación a violencia física → 3.' };
  if (/\b(que muera|muera en la c[aá]rcel|ojal[aá] se muera|que se le mueran|deseo que muera)\b/.test(t))
    return { ...out, a: '3', v: 'vw', r: 'Deseo de daño/muerte → 3.' };
  if (/\bmuerte!\b|\bmuerte\b/.test(t) && /\b(marero|pandillero|escoria)\b/.test(t))
    return { ...out, a: '3', v: 've', r: 'Exhortación/aprobación de muerte contra grupo → 3.' };
  if (/\b(exterminar|genocid|inferior(es)?|raza asquerosa|indios de mierda|malditos indios)\b/.test(t))
    return { ...out, a: '2', ir: 1, r: 'Ataque/desvalorización por raza/etnia → 2.' };
  if (/\b(musulman|jud[ií]|israel|palestin)\b/.test(t) && hasSlur)
    return { ...out, a: '2', irel: 1, r: 'Insulto grupal con base religiosa → 2.' };
  if (
    /\b(venezol|hondureñ|nicaragü|guatemalte|cuban|salvadoreñ|inmigrante|extranjer)\b/.test(t) &&
    /\b(mierda|basura|plaga|deport|devuelvan|parasit|fuera de|inferior|asqueros)\b/.test(t)
  )
    return { ...out, a: '2', io: 1, r: 'Exclusión/ataque por origen o nacionalidad → 2.' };
  if (/\b(fuera los|fuera de aqu[ií]).*(hondureñ|venezol|nicaragü|guatemalte|inmigrante)/.test(t))
    return { ...out, a: '2', io: 1, r: 'Exclusión por nacionalidad → 2.' };
  if (/\b(feminaz|mujeres de mierda|putas feministas|zorra|perra)\b/.test(t) && !/\b(mi perra|perros?)\b/.test(t))
    return { ...out, a: '2', ig: 1, r: 'Ataque sexista → 2.' };
  if (
    /\b(arena|fmln|nuevas ideas|bukele|oposici[oó]n)\b/.test(t) &&
    /\b(persecu|sin derechos|quitarles|excluir|deportar por apoyar)\b/.test(t)
  )
    return { ...out, a: '2', ip: 1, r: 'Exclusión/persecución por afiliación política → 2.' };
  if (/\b(pobres? de mierda|indigentes)\b/.test(t) && /\b(no merecen|sin derecho)\b/.test(t))
    return { ...out, a: '2', ic: 1, r: 'Negación de derechos por clase → 2.' };
  if (/\b(padre|madre|me dijeron|discrimin|sufr[ií]|me llamaron)\b/.test(t) && hasSlur)
    return { ...out, a: '0', qy: 1, ny: 1, r: 'Relato/cita de insulto ajeno o condena → 0.' };
  if (/\b(cerote|cerota|jueputa|vergaso|pendejo|culero|hijo de puta|malparido|mapachon|sorete|come mierda)\b/.test(t))
    return { ...out, a: '1', r: 'Insulto o vulgaridad dirigida → 1.' };
  if (/\b(independencia|#artesv|pintor salvadoreño|celebramos|orgullo salvadoreño|noticia|inform[oó])\b/.test(t) && !hasSlur)
    return { ...out, a: '0', r: 'Contenido cívico/informativo → 0.' };
  if (/\b(jajaja|messi|partido|tps|tr[aá]fico)\b/.test(t) && !hasSlur && !/\b(matar|muera)\b/.test(t))
    return { ...out, a: '0', r: 'Coloquial o crítica no tóxica → 0.' };
  if (/\b(maje|chero|cipote)\b/.test(t) && text.trim().length < 40)
    return {
      ...out,
      insuf: 1,
      ir: 'Jerga ambigua sin destinatario ni tono claro.',
      r: 'Contexto insuficiente; no forzar clase 0.',
    };
  if (text.trim().length < 18)
    return {
      ...out,
      insuf: 1,
      ir: 'Texto muy corto; falta contexto del acto.',
      r: 'Contexto insuficiente para etiqueta fiable.',
    };
  if (hasSlur)
    return { ...out, a: '1', review: 1, r: 'Vulgaridad presente; base identitaria no demostrable → 1 (revisar).' };
  return { ...out, a: '0', r: 'Sin ataque, odio ni violencia promovida → 0.' };
}
async function submit(dec) {
  const before = await getInfo();
  if (before.empty) return { ok: false, empty: true, before };
  await page.evaluate(() =>
    [...document.querySelectorAll('button,[role=button]')].find(b => b.textContent?.includes('Descartar'))?.click()
  );
  await page.waitForTime(110);
  await page.evaluate(() => {
    document.querySelector('form')?.scrollIntoView({ block: 'center' });
    window.scrollBy(0, 150);
  });
  if (dec.insuf) {
    await ck(I.si);
    await fill('insufficient_context_reason', dec.ir);
    await ck(I.qi);
    await fill('subclass', 'N/A');
    await ck(I.du);
    await ck(I.qunk);
    await ck(I.nunk);
    await ck(I.vn);
  } else {
    await ck(I.sa);
    await ck(I['l' + dec.a]);
    if (dec.io) await ck(I.io);
    if (dec.ig) await ck(I.ig);
    if (dec.ir) await ck(I.ir);
    if (dec.ip) await ck(I.ip);
    if (dec.ic) await ck(I.ic);
    if (dec.irel) await ck(I.irel);
    await fill('subclass', 'N/A');
    await ck(I.d);
    await ck(dec.qy ? I.qy : I.qn);
    await ck(dec.ny ? I.ny : I.nn);
    await ck({ vi: I.vi, vw: I.vw, vt: I.vt, ve: I.ve }[dec.v] || I.vn);
    await ck(dec.review ? I.qr : I.qa);
    await fill('insufficient_context_reason', 'N/A');
  }
  await fill('rationale', dec.r);
  await page.waitForTime(150);
  const dis = await page.evaluate(
    () =>
      [...document.querySelectorAll('button,[role=button]')].find(x => (x.textContent || '').includes('Enviar'))
        ?.disabled
  );
  if (dis) return { ok: false, before, dec, dis };
  await page.evaluate(() => {
    [...document.querySelectorAll('button,[role=button]')]
      .find(x => (x.textContent || '').includes('Enviar') && !x.disabled)
      ?.click();
  });
  await page.waitForTime(900);
  const after = await getInfo();
  return { ok: after.n > before.n || after.empty, before, after, dec };
}
const batch = [];
for (let k = 0; k < 8; k++) {
  const info = await getInfo();
  if (info.empty) break;
  const dec = decide(info.text);
  const r = await submit(dec);
  batch.push({
    counter: r.before?.counter,
    assigned: dec.insuf ? 'insuf' : dec.a,
    preview: r.before?.text?.slice(0, 72),
    ok: r.ok,
    next: r.after?.counter,
    review: !!dec.review,
  });
  if (!r.ok) break;
  if (r.after?.empty) break;
}
const q = await getInfo();
return {
  batch,
  queue: q.counter,
  remaining: q.empty ? 0 : q.total - q.n + 1,
  empty: q.empty,
};
