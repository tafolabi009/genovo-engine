#!/usr/bin/env python3
"""Regenerate the 6 missing files."""
import os

BASE = r'C:\Users\USER\Downloads\game_engine\engine'
total = 0

def W(rel, content):
    global total
    p = os.path.join(BASE, *rel.split('/'))
    with open(p, 'w', newline='\n') as f:
        f.write(content)
    n = content.count('\n') + 1
    total += n
    print(f"  {rel}: {n} lines")

# Read existing file content from gen_final.py for settings_system
# For the 5 large files, we read them from the existing Write tool content

# 1. gameplay/src/settings_system.rs - from gen_final.py content
settings_content = open(os.path.join(BASE, 'gen_final.py'), 'r').read()
# Extract the settings_system content
start = settings_content.find('"gameplay/src/settings_system.rs": """')
if start >= 0:
    start = settings_content.index('"""', start) + 3
    end = settings_content.index('"""', start)
    W('gameplay/src/settings_system.rs', settings_content[start:end])
else:
    print("ERROR: Could not find settings_system in gen_final.py")

# 2-6: The large render/physics files need to be regenerated
# These were originally written via Write tool but didn't persist

# For pbr_v2.rs - this was the biggest file (1593 lines)
# Let me create a streamlined but still comprehensive version

print(f"\nRecreating large files from scratch...")

# We need to create these files fresh. Let me write them all inline.

# pbr_v2.rs
W('render/src/pbr_v2.rs', open(os.path.join(BASE, 'gen_pbr_v2.py'), 'r').read() if os.path.exists(os.path.join(BASE, 'gen_pbr_v2.py')) else generate_pbr_v2())

def generate_pbr_v2():
    """This won't be called since we handle it differently below."""
    return ""

print(f"\nTotal lines: {total}")
print("Note: Need to handle pbr_v2 and other large files separately")
