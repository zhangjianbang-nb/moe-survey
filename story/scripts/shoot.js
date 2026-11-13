const path = require('path');
const puppeteer = require('/home/whu/workspace/servers/chatgpt-browser/node_modules/puppeteer-core');

(async () => {
  const browser = await puppeteer.launch({
    executablePath: '/snap/bin/chromium',
    headless: true,
    pipe: true,
    args: ['--no-sandbox', '--disable-gpu', '--hide-scrollbars']
  });

  const file = 'file://' + path.resolve(__dirname, '../index.html');
  const shots = path.resolve(__dirname, '../shots');

  // 桌面端：滚到第三章正文中段
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 980, deviceScaleFactor: 1 });
  await page.goto(file, { waitUntil: 'load' });
  await page.evaluate(() => {
    document.documentElement.style.scrollBehavior = 'auto';
    const el = document.querySelector('#ch3');
    window.scrollTo(0, el.offsetTop + 260);
  });
  await new Promise(r => setTimeout(r, 500));
  await page.screenshot({ path: path.join(shots, 'desktop-ch3.png') });

  // 桌面端：滚到第三章的笔记卡片
  await page.evaluate(() => {
    const el = document.querySelector('#ch3 aside.note');
    window.scrollTo(0, el.offsetTop - 200);
  });
  await new Promise(r => setTimeout(r, 400));
  await page.screenshot({ path: path.join(shots, 'desktop-note.png') });

  // 移动端
  const m = await browser.newPage();
  await m.setViewport({ width: 390, height: 844, deviceScaleFactor: 2 });
  await m.goto(file, { waitUntil: 'load' });
  await m.evaluate(() => {
    document.documentElement.style.scrollBehavior = 'auto';
    const el = document.querySelector('#ch3');
    window.scrollTo(0, el.offsetTop + 120);
  });
  await new Promise(r => setTimeout(r, 500));
  await m.screenshot({ path: path.join(shots, 'mobile-ch3.png') });

  // 暗色主题首屏
  const d = await browser.newPage();
  await d.setViewport({ width: 1440, height: 980, deviceScaleFactor: 1 });
  await d.goto(file, { waitUntil: 'load' });
  await d.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
  await new Promise(r => setTimeout(r, 400));
  await d.screenshot({ path: path.join(shots, 'dark-cover.png') });

  await browser.close();
  console.log('done');
})().catch(e => { console.error(e.message); process.exit(1); });
