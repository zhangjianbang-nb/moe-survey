const path = require('path');
const puppeteer = require('/home/whu/workspace/servers/chatgpt-browser/node_modules/puppeteer-core');

const file = 'file://' + path.resolve(__dirname, '../index.html');
let pass = 0, fail = 0;
function check(name, cond, detail) {
  if (cond) { pass++; console.log('PASS  ' + name + (detail ? '  [' + detail + ']' : '')); }
  else { fail++; console.log('FAIL  ' + name + (detail ? '  [' + detail + ']' : '')); }
}

(async () => {
  const browser = await puppeteer.launch({
    executablePath: '/snap/bin/chromium',
    headless: true, pipe: true,
    args: ['--no-sandbox', '--disable-gpu']
  });

  // ---------- 桌面端 ----------
  const d = await browser.newPage();
  await d.setViewport({ width: 1440, height: 980 });
  await d.goto(file, { waitUntil: 'load' });
  await new Promise(r => setTimeout(r, 300));

  const desk = await d.evaluate(() => {
    const toc = document.querySelector('nav.toc');
    const links = Array.from(document.querySelectorAll('.toc-link'));
    const introEls = Array.from(document.querySelectorAll('.special-intro'));
    return {
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      tocPos: getComputedStyle(toc).position,
      tocWidth: toc.getBoundingClientRect().width,
      linkCount: links.length,
      deadLinks: links.filter(a => !document.querySelector(a.getAttribute('href'))).length,
      chapters: document.querySelectorAll('section.chapter').length,
      hasPreface: !!document.querySelector('#preface'),
      hasEpilogue: !!document.querySelector('#epilogue'),
      hasAppendix: !!document.querySelector('#appendix'),
      hasGlossary: !!document.querySelector('#glossary'),
      tableRows: document.querySelectorAll('table.numbers tbody tr').length,
      glossaryTerms: document.querySelectorAll('dl.glossary dt').length,
      introCount: introEls.length,
      introInChapters: introEls.filter(el => el.closest('section.chapter')).length,
      introLeadFont: getComputedStyle(document.querySelector('.special-intro .si-lead')).fontSize,
      introUlFont: getComputedStyle(document.querySelector('.special-intro ul')).fontSize,
      bodyBg: getComputedStyle(document.body).backgroundColor,
      fontFamily: getComputedStyle(document.body).fontFamily.split(',')[0],
      fontSize: getComputedStyle(document.body).fontSize,
      lineHeight: getComputedStyle(document.body).lineHeight,
      progressBar: !!document.getElementById('progress'),
    };
  });

  check('桌面无横向溢出', desk.overflow <= 0, 'overflow=' + desk.overflow + 'px');
  check('桌面目录 sticky', desk.tocPos === 'sticky', desk.tocPos);
  check('桌面目录宽度合理', desk.tocWidth >= 200 && desk.tocWidth <= 260, desk.tocWidth.toFixed(0) + 'px');
  check('目录链接数 10', desk.linkCount === 10, desk.linkCount);
  check('目录无死链', desk.deadLinks === 0, desk.deadLinks + ' dead');
  check('章节数 6', desk.chapters === 6, desk.chapters);
  check('序章存在', desk.hasPreface);
  check('尾声存在', desk.hasEpilogue);
  check('附录存在', desk.hasAppendix);
  check('术语小抄存在', desk.hasGlossary);
  check('数字表 21 行', desk.tableRows === 21, desk.tableRows);
  check('术语 17 条', desk.glossaryTerms === 17, desk.glossaryTerms);
  check('专业导读 7 篇（序 1 + 章 6）', desk.introCount === 7 && desk.introInChapters === 6,
        desk.introCount + ' total / ' + desk.introInChapters + ' in chapters');
  check('专业导读字号层级正确', parseFloat(desk.introLeadFont) > parseFloat(desk.introUlFont),
        desk.introLeadFont + ' vs ' + desk.introUlFont);
  check('正文行高 >= 1.8', parseFloat(desk.lineHeight) / parseFloat(desk.fontSize) >= 1.8,
        (parseFloat(desk.lineHeight) / parseFloat(desk.fontSize)).toFixed(2));
  check('正文衬线字体', /Georgia|Songti|serif|Song/i.test(desk.fontFamily), desk.fontFamily);
  check('进度条存在', desk.progressBar);

  // 暗色对比度
  await d.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
  await new Promise(r => setTimeout(r, 600)); // 等 0.35s 背景过渡结束再采样
  const dark = await d.evaluate(() => {
    function lum(c) {
      const m = c.match(/\d+/g).map(Number).slice(0, 3).map(v => {
        v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
      });
      return 0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2];
    }
    const bodyBg = getComputedStyle(document.body).backgroundColor;
    const ink = getComputedStyle(document.body).color;
    const note = document.querySelector('aside.note');
    const noteBg = getComputedStyle(note).backgroundColor;
    const si = document.querySelector('.special-intro');
    const siBg = getComputedStyle(si).backgroundColor;
    const siLead = getComputedStyle(si.querySelector('.si-lead')).color;
    const r = (lum(ink) + 0.05) / (lum(bodyBg) + 0.05);
    const rsi = (lum(siLead) + 0.05) / (lum(siBg) + 0.05);
    return { bodyBg, ink, noteBg, siBg, siLead, contrast: r, siContrast: rsi };
  });
  check('暗色正文对比度 >= 7:1 (AAA)', dark.contrast >= 7, dark.contrast.toFixed(1) + ':1');
  check('暗色卡片背景已提亮', dark.noteBg !== dark.bodyBg, dark.noteBg + ' vs ' + dark.bodyBg);
  check('暗色专业导读对比度 >= 7:1', dark.siContrast >= 7, dark.siContrast.toFixed(1) + ':1');

  // ---------- 移动端 ----------
  const m = await browser.newPage();
  await m.setViewport({ width: 390, height: 844 });
  await m.goto(file, { waitUntil: 'load' });
  await new Promise(r => setTimeout(r, 300));

  const mob = await m.evaluate(() => {
    const toc = document.querySelector('nav.toc');
    const ol = toc.querySelector('ol');
    const firstLink = toc.querySelector('a');
    const lead = document.querySelector('section.chapter p.lead');
    const tableWrap = document.querySelector('.table-wrap');
    const overflow = document.documentElement.scrollWidth - document.documentElement.clientWidth;
    const cs = getComputedStyle(lead, '::first-letter');
    return {
      overflow,
      tocOverflowX: getComputedStyle(toc).overflowX,
      olDisplay: getComputedStyle(ol).display,
      linkHeight: firstLink.getBoundingClientRect().height,
      linkRadius: getComputedStyle(firstLink).borderRadius,
      firstLetterSize: cs.fontSize,
      firstLetterFloat: cs.float,
      bodyFontSize: getComputedStyle(document.body).fontSize,
      tableWrapOverflow: getComputedStyle(tableWrap).overflowX,
      siPadding: getComputedStyle(document.querySelector('.special-intro')).padding,
    };
  });
  check('移动端无横向溢出', mob.overflow <= 0, 'overflow=' + mob.overflow + 'px');
  check('移动端目录横向滚动', mob.tocOverflowX === 'auto', mob.tocOverflowX);
  check('移动端目录 flex 胶囊', mob.olDisplay === 'flex' && mob.linkRadius === '999px',
        mob.olDisplay + ' / radius ' + mob.linkRadius);
  check('移动端首字下沉已禁用', mob.firstLetterFloat === 'none' && parseFloat(mob.firstLetterSize) <= 20,
        'float=' + mob.firstLetterFloat + ', size=' + mob.firstLetterSize);
  check('移动端表格可横滚', mob.tableWrapOverflow === 'auto', mob.tableWrapOverflow);
  check('移动端专业导读收紧内边距', mob.siPadding.includes('18px'), mob.siPadding);

  await browser.close();
  console.log('\n==== ' + pass + ' PASS / ' + fail + ' FAIL ====');
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error('ERR', e.message); process.exit(2); });
