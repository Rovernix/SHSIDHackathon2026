// Optional UI QA. Set PLAYWRIGHT_MODULE to a Playwright installation if necessary.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const {pathToFileURL} = require('url');

(async()=>{
  const root=path.resolve(__dirname,'..');
  const input=JSON.parse(fs.readFileSync(path.join(root,'results','main','input.json'),'utf8'));
  const baseline=JSON.parse(fs.readFileSync(path.join(root,'results','main','fifo.json'),'utf8'));
  const fmt=(x)=>Number(x).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
  const out=path.join(root,'results','qa','screenshots');fs.mkdirSync(out,{recursive:true});
  const browser=await chromium.launch({channel:process.env.AIDC_BROWSER_CHANNEL || 'chrome',headless:true});
  const evidence=[];
  for(const width of [1280,390]){
    const page=await browser.newPage({viewport:{width,height:900}});const errors=[];
    page.on('pageerror',e=>errors.push(String(e)));
    await page.goto(pathToFileURL(path.join(root,'output','demo','Results_Explorer.html')).href);
    await page.waitForFunction(()=>window.__RESULTS_READY__===true);
    const body=await page.locator('body').innerText();
    if(body.includes('IMMC') || body.includes('Published reference appendix'))throw new Error('Removed supplemental content remains in the explorer');
    if(body.includes('50 kW grid scenario') || /__[A-Z_]+__/.test(body))throw new Error('Stale scenario text or unresolved template field');
    if(!body.includes('both grid and renewable supply to zero'))throw new Error('Impossible-supply explanation differs from the saved stress');
    for(const strategy of ['joint_green','joint_cost','compute_only','fifo_storage','fifo','rolling']){
      await page.selectOption('#strategy',strategy);
      const card=await page.locator('#cards strong').allTextContents();
      if(card.length!==4 || card.some(v=>v.includes('NaN')))throw new Error('Invalid metrics '+strategy);
      const saved=JSON.parse(fs.readFileSync(path.join(root,'results','main',strategy+'.json'),'utf8'));
      const expected=[fmt(saved.cost_cny),fmt(100*(1-saved.cost_cny/baseline.cost_cny))+'%',fmt(100*(1-saved.carbon_kg/baseline.carbon_kg))+'%',fmt(saved.peak_kw)];
      if(card.some((value,i)=>value!==expected[i]))throw new Error('Metrics differ from saved executable result '+strategy);
      if(strategy==='joint_green'){
        const expectedGap=Number(saved.mip_gap*100).toLocaleString('en',{minimumFractionDigits:6,maximumFractionDigits:6})+'%';
        if(!(await page.locator('#certificate').innerText()).includes(expectedGap))throw new Error('Certificate hides the saved gap');
      }
      if(strategy==='rolling' && !(await page.locator('#policy').innerText()).includes('previous-day persistence'))throw new Error('Online forecast description is stale');
      if(await page.locator('#jobs tr').count()!==input.jobs.length)throw new Error('Dropped job rows');
      await page.locator('#slot').fill('95');await page.locator('#slot').dispatchEvent('input');
      if(!(await page.locator('#clock').innerText()).includes('24:00'))throw new Error('Terminal slot missing');
      await page.locator('#search').fill(input.jobs[4].id);
      if(await page.locator('#jobs tr').count()!==1)throw new Error('Search broken');
      await page.locator('#search').fill('');
      const layout=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));
      if(layout.scroll>layout.width+1)throw new Error('Viewport overflow '+width+' '+strategy);
      evidence.push({width,strategy,metrics:card,jobRows:input.jobs.length,terminalSlot:true,search:true,viewportOverflow:false});
    }
    await page.selectOption('#strategy','joint_green');
    await page.locator('#slot').fill('48');await page.locator('#slot').dispatchEvent('input');
    await page.screenshot({path:path.join(out,`explorer-${width}.png`),fullPage:true});
    if(errors.length)throw new Error(errors.join('\n'));
    await page.close();
  }
  await browser.close();
  const explorer_sha256=crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'output','demo','Results_Explorer.html'))).digest('hex');
  const template_sha256=crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'templates','results_explorer.html'))).digest('hex');
  fs.writeFileSync(path.join(root,'results','qa','browser_qa.json'),JSON.stringify({passed:true,cases:evidence,explorer_sha256,template_sha256,documentationAssertions:true},null,2));
  process.stdout.write(`Passed ${evidence.length} browser cases.\n`);
})().catch(e=>{process.stderr.write(String(e.stack||e));process.exit(1)});
