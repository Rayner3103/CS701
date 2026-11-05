#!/usr/bin/env python3
"""One-off converter from the archived LaTeX report to Markdown, and table fragments for gen_report.py."""
import re
SYM = {'alpha':'α','gamma':'γ','rho':'ρ','Delta':'Δ','sigma':'σ','times':'×','cdot':'·','ge':'≥','le':'≤','gg':'≫','approx':'≈','leftarrow':'←','rightarrow':'→','pi':'π','theta':'θ','mid':'|','neq':'≠','quad':' ','ldots':'...','dots':'...','lambda':'λ','pm':'±','sim':'∼','ne':'≠','sum':'Σ','leq':'≤','geq':'≥','log':'log','max':'max','arg':'arg','exp':'exp'}
MAC = {'GPUH':'TBD','textdegree':'°','y':'','mn':'−','x':'×','pmm':'±','ap':'≈','to':'→','sg':'σ','lam':'λ','dm':'Δm','sn':'†','dc':'‡','vtp':"v3′",'ldots':'...'}
def arg(s, i):
    """s[i] == '{': return (content, index after closing brace)."""
    d = 0
    for j in range(i, len(s)):
        if s[j] == '\\': continue
        if s[j] == '{' and (j == 0 or s[j - 1] != '\\'): d += 1
        if s[j] == '}' and s[j - 1] != '\\':
            d -= 1
            if d == 0: return s[i + 1:j], j + 1
    raise ValueError(s[i:i + 80])
def sub_cmd(s, name, fn):
    pat = '\\' + name + '{'
    while True:
        i = s.find(pat)
        if i < 0: return s
        c, e = arg(s, i + len(pat) - 1)
        s = s[:i] + fn(c) + s[e:]
class Conv:
    def __init__(self): self.refs = {'sec:val': '4.1', 'sec:findings': '4.5', 'sec:abl': '4.3', 'sec:terms': '5', 'sec:method': '3', 'tab:loo': '2', 'tab:pp': '3', 'tab:official': '1', 'tab:neg': '4', 'tab:terms': '5'}; self.code = []
    def protect(self, t): self.code.append(t); return f'\x00{len(self.code) - 1}\x00'
    def math(self, m):
        m = m.replace('\\{', '\x01').replace('\\}', '\x02').replace('\\_', '_')
        m = re.sub(r'\\text\{([^}]*)\}', lambda k: ' \x03' + k.group(1) + '\x03 ', m)
        m = re.sub(r'\\sqrt\{([^}]*)\}', r'√\1', m); m = re.sub(r'\\overline\{([^}]*)\}', r'mean(\1)', m); m = re.sub(r'\\bar\s*\\?([A-Za-z]+)', lambda k: 'mean(' + SYM.get(k.group(1), k.group(1)) + ')', m)
        m = re.sub(r'\\([A-Za-z]+)', lambda k: ' ' + SYM.get(k.group(1), MAC.get(k.group(1), k.group(1))) + ' ', m)
        m = m.replace('\\,', ' ').replace('\\ ', ' ').replace('{', '').replace('}', '').replace('\x01', '{').replace('\x02', '}')
        m = re.sub(r' ?\x03 ?', '', m)
        m = re.sub(r' {2,}', ' ', m).strip()
        m = re.sub(r' ?([=<>≤≥≠←→×·+]) ?', r' \1 ', m); m = re.sub(r'\( ', '(', m); m = re.sub(r' \)', ')', m); m = re.sub(r'\| ', '|', m)
        m = re.sub(r' ?([_^]) ?', r'\1', m); m = m.replace('arg max', 'argmax'); m = re.sub(r'\b(max|min|exp|cos|log|arg) \(', r'\1(', m); m = re.sub(r'π \(', 'π(', m); m = re.sub(r'/ ', '/', m)
        return m
    def inline(self, s):
        s = sub_cmd(s, 'fl', lambda c: self.protect('`' + c.replace('|', '\\|') + '`'))
        s = sub_cmd(s, 'texttt', lambda c: self.protect('`' + c.replace('\\_', '_').replace('\\%', '%').replace('|', '\\|') + '`'))
        s = s.replace('\\S\\ref', '§\\ref').replace('\\S~\\ref', '§\\ref')
        s = re.sub(r'\\ref\{([^}]*)\}', lambda m: self.refs.get(m.group(1), m.group(1)), s)
        s = re.sub(r'(?:Table|Tables|Figure)~?(?=[AT F]?[A-Z]?\d)', lambda m: m.group(0).replace('~', ' '), s)
        s = re.sub(r'\$([^$]*)\$', lambda m: self.math(m.group(1)), s)
        s = sub_cmd(s, 'textbf', lambda c: '**' + c.strip() + '**'); s = sub_cmd(s, 'emph', lambda c: '*' + c.strip() + '*')
        s = sub_cmd(s, 'textcolor{red}', lambda c: '**' + c + '**'); s = sub_cmd(s, 'ensuremath', lambda c: self.math(c))
        s = re.sub(r'\\cite\{([^}]*)\}', lambda m: '[' + ', '.join(m.group(1).split(',')) + ']', s)
        s = re.sub(r'\\([a-zA-Z]+)\{\}', lambda m: MAC.get(m.group(1), m.group(0)), s)
        s = re.sub(r'\\S\b ?', '§', s)
        s = re.sub(r'\\([a-zA-Z]+)( ?)', lambda m: MAC[m.group(1)] + m.group(2) if m.group(1) in MAC else (SYM[m.group(1)] + m.group(2) if m.group(1) in SYM else m.group(0)), s)
        for a, b in (('\\_', '_'), ('\\%', '%'), ('\\&', '&'), ('\\#', '#'), ('\\$', '$'), ('``', '"'), ("''", '"'), ('---', '—'), ('--', '–'), ('~', ' '), ('\\,', ' '), ('\\ ', ' '), ('{}', ''), ('\\\\', ' ')): s = s.replace(a, b)
        s = re.sub(r'\x00(\d+)\x00', lambda m: self.code[int(m.group(1))], s)
        return s.strip()
    def row(self, r):
        r = r.strip()
        r = re.sub(r'\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$', lambda m: m.group(2) + ' & ' * (int(m.group(1)) - 1), r)
        cells = re.split(r'(?<!\\)&', r)
        return '| ' + ' | '.join(self.inline(c).replace('\n', ' ') or ' ' for c in cells) + ' |'
    def table(self, head, rows, caption, num):
        h = self.row(head); n = h.count(' | ') + 1
        out = [f'**{num}.** {self.inline(caption)}', '', h, '|' + '---|' * n]
        for r in rows:
            m = self.row(r); k = m.count(' | ') + 1
            out.append(m if k >= n else m[:-2] + ' | ' * (n - k) + ' |')
        return '\n'.join(out)
def rows_of(body):
    body = re.sub(r'\\(toprule|midrule|bottomrule|endhead|endfirsthead)\b', '', body)
    parts = re.split(r'\\\\(?:\[[^\]]*\])?\s*\n?', body)
    return [p.strip() for p in parts if p.strip()]
def tex_table_to_md(tex, num, conv=None):
    """A generated \\begin{table}...\\end{table} from gen_report.py -> Markdown."""
    conv = conv or Conv()
    cap, _ = arg(tex, tex.index('\\caption{') + 8)
    b = tex[tex.index('\\begin{tabular}'):tex.index('\\end{tabular}')]
    b = b[b.index('\n') + 1:]
    r = rows_of(b)
    return conv.table(r[0], r[1:], cap.replace('\\S\\ref{sec:val}', '§4.1').replace('Table~\\ref{tab:loo}', 'Table 2'), num)
def convert_body(src, c, prefix):
    """Section numbers and table numbers are assigned in a first pass, so forward references resolve."""
    sec = sub = 0; tn = 0
    for m in re.finditer(r'\\(section|subsection|subsubsection)\{|\\label\{([^}]*)\}|\\begin\{lt\}', src):
        if m.group(1) == 'section': sec += 1; sub = 0; cur = f'{prefix}{sec}'
        elif m.group(1) == 'subsection': sub += 1; cur = f'{prefix}{sec}.{sub}'
        elif m.group(2):
            if m.group(2).startswith('tab:'): c.refs[m.group(2)] = f'{prefix}{tn}' if False else c.refs.get(m.group(2), '')
            else: c.refs[m.group(2)] = cur
        elif m.group(0).startswith('\\begin{lt}'): tn += 1; tl = re.search(r'\\label\{(tab:[^}]*)\}', src[m.end():m.end() + 900]);  c.refs[tl.group(1)] = f'{tn}' if False else f'{prefix}{tn}' if tl else ''
    return md_lines(src, c, prefix)
def md_lines(src, c, prefix):
    out = []; i = 0; L = src.split('\n'); sec = sub = 0; tn = 0; para = []
    def flush():
        if para: out.append(c.inline(' '.join(para))); out.append(''); para.clear()
    while i < len(L):
        l = L[i].rstrip()
        s = l.strip()
        if s.startswith('%') or s in ('{\\footnotesize', '}', '\\newpage', '\\clearpage'): i += 1; continue
        if s == '': flush(); i += 1; continue
        m = re.match(r'\\(section|subsection|subsubsection)\*?\{(.*)\}$', s)
        if m:
            flush()
            if m.group(1) == 'section': sec += 1; sub = 0; out += [f'## {prefix}{sec}. {c.inline(m.group(2))}', '']
            elif m.group(1) == 'subsection': sub += 1; out += [f'### {prefix}{sec}.{sub} {c.inline(m.group(2))}', '']
            else: out += [f'#### {c.inline(m.group(2))}', '']
            i += 1; continue
        if re.fullmatch(r'\\label\{[^}]*\}', s): i += 1; continue
        if s.startswith('\\input{'):  i += 1; continue
        if s.startswith('\\begin{verbatim}'):
            flush(); out.append('```bash'); i += 1
            while not L[i].strip().startswith('\\end{verbatim}'): out.append(L[i]); i += 1
            out += ['```', '']; i += 1; continue
        if s.startswith('\\begin{align*}'):
            flush(); j = i + 1; b = []
            while not L[j].strip().startswith('\\end{align*}'): b.append(L[j]); j += 1
            out += ['```text'] + [c.math(x.replace('&', '').replace('\\\\', '')) for x in b] + ['```', '']; i = j + 1; continue
        if s.startswith('\\begin{itemize}') or s.startswith('\\begin{enumerate}'):
            flush(); num = 'enumerate' in s; n = 0; i += 1; cur = None
            while not L[i].strip().startswith(('\\end{itemize}', '\\end{enumerate}')):
                t = L[i].strip()
                if t.startswith('\\item'):
                    if cur is not None: n += 1; out.append((f'{n}. ' if num else '- ') + c.inline(cur))
                    cur = t[5:].strip()
                elif t: cur = (cur or '') + ' ' + t
                i += 1
            if cur is not None: n += 1; out.append((f'{n}. ' if num else '- ') + c.inline(cur))
            out.append(''); i += 1; continue
        if s.startswith('\\begin{lt}'):
            flush(); j = i + 1; b = []
            while not L[j].strip().startswith('\\end{lt}'): b.append(L[j]); j += 1
            t = '\n'.join(b); cap, e = arg(t, t.index('\\caption{') + 8); h, e2 = arg(t, t.index('\\lthead{') + 7 + len('{') + t[t.index('\\lthead{') + 8:].index('{') + 1 + 0) if False else (None, None)
            k = t.index('\\lthead{'); _, e1 = arg(t, k + 7); h, e2 = arg(t, e1)
            tn += 1; out += [c.table(h, rows_of(t[e2:]), cap, f'Table {prefix}{tn}'), '']; i = j + 1; continue
        para.append(s); i += 1
    flush()
    return '\n'.join(out)
