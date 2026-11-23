import assert from 'node:assert/strict'
import { renderMarkdownLite } from './markdown-lite.mjs'

const html = renderMarkdownLite('## 概述\n\n- 工业结构健康监测\n- 远程生理信号检测\n\n<script>alert(1)</script>')

assert.match(html, /<h3>概述<\/h3>/, 'level-2 Markdown headings should render as h3')
assert.match(html, /<ul><li>工业结构健康监测<\/li><li>远程生理信号检测<\/li><\/ul>/, 'dash lists should render as unordered lists')
assert.doesNotMatch(html, /## 概述/, 'raw Markdown heading markers should not remain visible')
assert.doesNotMatch(html, /<script>/, 'raw HTML should stay escaped')
assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/, 'escaped HTML text should remain readable')
assert.match(renderMarkdownLite("'quoted'"), /&#39;quoted&#39;/, 'single quotes should stay escaped')

const unsafeLinks = renderMarkdownLite('[x](javascript:alert(1))\n[y](JaVaScRiPt:alert(1))')
assert.doesNotMatch(unsafeLinks, /<a\b/, 'javascript URLs should render as text without an anchor')
assert.equal(unsafeLinks, '<p>x<br>y</p>', 'unsupported links should leave only their link text')

const httpsLink = renderMarkdownLite('[docs](https://example.com/path)')
assert.match(httpsLink, /<a href="https:\/\/example\.com\/path" target="_blank" rel="noopener">docs<\/a>/, 'https links should still render')

const quoteBreakout = renderMarkdownLite('[x](https://example.com/" onmouseover="alert(1))')
const linkTag = quoteBreakout.match(/<a\b[^>]*>/)?.[0]
assert.equal(
  linkTag,
  '<a href="https://example.com/&quot; onmouseover=&quot;alert(1)" target="_blank" rel="noopener">',
  'quotes in the URL must stay escaped inside the href attribute',
)
assert.match(linkTag, /^<a href="[^"]*" target="_blank" rel="noopener">$/, 'the generated anchor must not add an onmouseover attribute')
