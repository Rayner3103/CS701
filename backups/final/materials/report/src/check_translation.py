#!/usr/bin/env python3
"""check_translation.py EN.md ZH.md: every number, inline code span and table shape in EN must survive in ZH."""
import re, sys, collections
en, zh = open(sys.argv[1]).read(), open(sys.argv[2]).read()
num = lambda s: collections.Counter(re.findall(r'\d+(?:[.,]\d+)*', re.sub(r'!\[[^\]]*\]\([^)]*\)', '', s)))
code = lambda s: collections.Counter(re.findall(r'`[^`\n]+`', s))
a, b = num(en), num(zh); print('numbers missing:', dict(list((a - b).items())[:25]), '| extra:', dict(list((b - a).items())[:25]))
a, b = code(en), code(zh); print('code spans missing:', list((a - b).items())[:15])
ea, za = en.split('\n'), zh.split('\n')
shape = lambda L: [l.count(' | ') for l in L if l.startswith('| ')]
print('table rows', len(shape(ea)), len(shape(za)), 'shape equal', shape(ea) == shape(za)); print('headings', sum(l.startswith('#') for l in ea), sum(l.startswith('#') for l in za), 'markers', en.count('<!--TABLE'), zh.count('<!--TABLE'), 'zh images', zh.count('.zh-CN.png'), 'en images', en.count('.png)'))
