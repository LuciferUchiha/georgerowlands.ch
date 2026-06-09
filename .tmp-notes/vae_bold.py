import re

path = r'c:\Users\George\Workspace\georgerowlands.ch\content\garden\ml\computerVision\autoencoders.md'
text = open(path, encoding='utf-8').read()

START = '## What Makes a Good Representation?'
i = text.index(START)
head, region = text[:i], text[i:]

mask_re = re.compile(
    r'\\begin\{[^}]*\}'
    r'|\\end\{[^}]*\}'
    r'|\\(?:text|mathrm|mathbb|mathcal|mathbf|mathsf|operatorname|boldsymbol)\{[^{}]*\}'
    r'|\\[a-zA-Z]+'
)
xz_re = re.compile(r'(?<!_)(x|z)(?!_j)')

def process_math(s):
    # greek vectors (bold via \boldsymbol; KaTeX \mathbf does not bold lowercase greek)
    s = s.replace(r'\mu_\phi', r'\boldsymbol{\mu}_\phi')
    s = s.replace(r'\mu_\theta', r'\boldsymbol{\mu}_\theta')
    s = s.replace(r'\sigma_\phi', r'\boldsymbol{\sigma}_\phi')
    s = s.replace(r'\sigma_\theta', r'\boldsymbol{\sigma}_\theta')
    s = s.replace(r'\mathcal{N}(\mu,', r'\mathcal{N}(\boldsymbol{\mu},')  # standalone mean vector
    s = re.sub(r'\\epsilon(?!_j)', r'\\boldsymbol{\\epsilon}', s)        # epsilon vector, not component
    s = s.replace(' I)', r' \mathbf{I})')                               # identity covariance
    s = s.replace(r'\hat{x}', r'\hat{\mathbf{x}}')
    # bold latin x, z (mask commands first so densities/words are untouched)
    store = []
    def stash(m):
        store.append(m.group(0))
        return '\x00%d\x00' % (len(store) - 1)
    masked = mask_re.sub(stash, s)
    masked = xz_re.sub(lambda m: r'\mathbf{%s}' % m.group(1), masked)
    return re.sub('\x00(\\d+)\x00', lambda m: store[int(m.group(1))], masked)

region = re.sub(r'\$\$.*?\$\$|\$[^$]*?\$', lambda m: process_math(m.group(0)), region, flags=re.DOTALL)
open(path, 'w', encoding='utf-8').write(head + region)
print('done')
