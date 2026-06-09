import re

path = r'c:\Users\George\Workspace\georgerowlands.ch\content\garden\ml\computerVision\autoencoders.md'
text = open(path, encoding='utf-8').read()

START = '## Linear Autoencoders'
END = '## What Makes a Good Representation?'
i = text.index(START)
j = text.index(END)
head, lin, tail = text[:i], text[i:j], text[j:]

MATS = ['W', 'V', 'P', 'Q', 'A', 'S', 'B', 'C', 'I']   # matrices (U handled manually)
VECS = ['x', 'z', 'u', 'v', 'e', 'q', 'y', 'a', 'b', 'c', 'w']  # vectors
TARGETS = MATS + VECS
target_re = re.compile(r'(?<!_)(' + '|'.join(TARGETS) + r')')

# mask commands and braced-name groups so their letters are not bolded
mask_re = re.compile(
    r'\\begin\{[^}]*\}'
    r'|\\end\{[^}]*\}'
    r'|\\(?:text|mathrm|mathbb|mathcal|mathbf|mathsf|operatorname)\{[^{}]*\}'
    r'|\\[a-zA-Z]+'
)

def bold_segment(s):
    store = []
    def stash(m):
        store.append(m.group(0))
        return '\x00%d\x00' % (len(store) - 1)
    masked = mask_re.sub(stash, s)
    masked = target_re.sub(lambda m: r'\mathbf{%s}' % m.group(1), masked)
    return re.sub('\x00(\\d+)\x00', lambda m: store[int(m.group(1))], masked)

lin = re.sub(r'\$\$.*?\$\$|\$[^$]*?\$', lambda m: bold_segment(m.group(0)), lin, flags=re.DOTALL)

open(path, 'w', encoding='utf-8').write(head + lin + tail)
print('done')
