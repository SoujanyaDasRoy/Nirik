#!/usr/bin/env python3
import sys

file_path = "/c/Users/sdroy/OneDrive/Desktop/Documents/Final Year Project/backend/core/inference.py"
with open(file_path, 'r') as f:
    lines = f.readlines()

# Replace lines 93-104 (0-indexed 92-103) with correct pad_to_square
new_lines = lines[:92]  # up to line 92 exclusive (line numbers start at 1)
new_lines.append('def pad_to_square(img: Image.Image, fill=0) -> Image.Image:\n')
new_lines.append('    w, h = img.size\n')
new_lines.append('    if w == h:\n')
new_lines.append('        return img\n')
new_lines.append('    elif w > h:\n')
new_lines.append('        result = Image.new(img.mode, (w, w), fill)\n')
new_lines.append('        result.paste(img, (0, (w - h) // 2))\n')
new_lines.append('        return result\n')
new_lines.append('    else:\n')
new_lines.append('        result = Image.new(img.mode, (h, h), fill)\n')
new_lines.append('        result.paste(img, ((h - w) // 2, 0))\n')
new_lines.append('        return result\n')
# Append the rest of the lines after line 104 (index 103)
new_lines.extend(lines[104:])  # from line 105 onwards (0-index)

with open(file_path, 'w') as f:
    f.writelines(new_lines)

print("Fixed pad_to_square function.")