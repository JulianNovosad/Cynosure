with open('agent1_log.txt', 'r') as f:
    lines = f.readlines()

chars = []
for line in lines:
    start = line.index('"') + 1
    end = line.index('"', start)
    char = line[start:end]
    chars.append(char)

transitions = []
prev = None
for i, c in enumerate(chars):
    if c != prev:
        tick_line = lines[i].strip()
        transitions.append(tick_line)
        prev = c

print('CHARACTER TRANSITIONS:')
for t in transitions:
    print(t)

print()
print('UNIQUE CHARACTERS IN ORDER OF APPEARANCE:')
seen = []
for c in chars:
    if c not in seen:
        seen.append(c)
print(' -> '.join(repr(c) for c in seen))

print()
print('CHARACTER FREQUENCY:')
from collections import Counter
freq = Counter(chars)
for char, count in freq.most_common():
    print(f'  {repr(char):10s}: {count:5d} ticks ({count/len(chars)*100:.1f}%)')

print()
print(f'TOTAL AGENT 1 WHISPER EVENTS: {len(chars)}')
