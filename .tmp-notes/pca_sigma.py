import re

path = r'c:\Users\George\Workspace\georgerowlands.ch\content\garden\maths\linearAlgebra\pca.md'
text = open(path, encoding='utf-8').read()

def fix(s):
    # within a math segment, the only standalone capital S is the covariance matrix
    # S followed by a letter (matrix-vector product like Sv, Su_i) -> "\Sigma " (keep space)
    s = re.sub(r'(?<![A-Za-z\\])S(?=[A-Za-z])', r'\\Sigma ', s)
    # S followed by non-letter (S_, S=, (S), S^ ...) -> "\Sigma"
    s = re.sub(r'(?<![A-Za-z\\])S(?![A-Za-z])', r'\\Sigma', s)
    return s

text = re.sub(r'\$\$.*?\$\$|\$[^$]*?\$', lambda m: fix(m.group(0)), text, flags=re.DOTALL)
open(path, 'w', encoding='utf-8').write(text)
print('done')
