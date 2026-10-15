// 语法检查：抽取 dashboard.html 中真正的 JS <script> 块，用 new Function 解析（不执行）
// 跳过：外部脚本（src=）、JSON 数据岛（type=application/json 等）、模板型 type
const fs = require('fs');
const nodePath = require('path');
// 无参数时默认校验项目内的 app/dashboard.html（脚本可位于项目根或 scripts/ 下）
const filePath = process.argv[2] || nodePath.join(__dirname, '..', 'app', 'dashboard.html');
const html = fs.readFileSync(filePath, 'utf8');
const JS_TYPES = ['', 'text/javascript', 'application/javascript', 'module'];
const blocks = [...html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)]
  .map(m => ({ attrs: m[1] || '', code: m[2] }))
  .filter(b => !/\bsrc\s*=/.test(b.attrs))
  .filter(b => {
    const t = (b.attrs.match(/\btype\s*=\s*["']([^"']*)["']/) || [, ''])[1].trim().toLowerCase();
    return JS_TYPES.includes(t);
  });
let ok = 0, fail = 0;
blocks.forEach((b, i) => {
  try {
    new Function(b.code);          // 只解析，不执行
    ok++;
    console.log(`script#${i} OK  (${b.code.length} chars)`);
  } catch (e) {
    fail++;
    console.log(`script#${i} SYNTAX ERROR: ${e.message}`);
  }
});
console.log(`\n${blocks.length} JS script block(s): ${ok} ok, ${fail} failed`);
process.exit(fail ? 1 : 0);
