import re

path = r'c:\Users\George\Workspace\georgerowlands.ch\content\garden\ml\computerVision\autoencoders.md'
text = open(path, encoding='utf-8').read()

# 1. covariance matrix S -> Sigma (remaining occurrences in PCA theorem, learning algos, etc.)
text = text.replace(r'\mathbf{S}', r'\mathbf{\Sigma}')

# 2. bracket bare expectations of a squared norm: \mathbb{E}\|...\|^2 -> \mathbb{E}\big[\|...\|^2\big]
#    (does not touch already-bracketed \mathbb{E}\big[... since those are not followed immediately by \|)
text = re.sub(r'\\mathbb\{E\}\\\|(.+?)\\\|\^2', r'\\mathbb{E}\\big[\\|\1\\|^2\\big]', text)

# 3. bracket bare \mathbb{E}\mathbf{x} -> \mathbb{E}[\mathbf{x}]
text = text.replace(r'\mathbb{E}\mathbf{x}', r'\mathbb{E}[\mathbf{x}]')

open(path, 'w', encoding='utf-8').write(text)
print('done')
