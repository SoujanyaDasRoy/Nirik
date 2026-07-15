#!/usr/bin/env python3
import sys

file_path = "/c/Users/sdroy/OneDrive/Desktop/Documents/Final Year Project/backend/explainability/roi_extraction.py"
with open(file_path, 'r') as f:
    lines = f.readlines()

# Fix line 241 (0-indexed 240)
# Original broken line: roi_count = evidence.get('roi_metrics', {}).getroi_count)customer.get('count', 0)
# Should be: roi_count = evidence.get('roi_metrics', {}).get('count', 0)
lines[240] = "    roi_count = evidence.get('roi_metrics', {}).get('count', 0)\n"

# Fix line 316 (0-indexed 315)
# Original: reasoning_confidence = (evid_conf * 0.6 + conf_conf * 0.4)
# Should be: reasoning_confidence = (evid_conf * 0.6 + cal_conf * 0.4)
lines[315] = "    reasoning_confidence = (evid_conf * 0.6 + cal_conf * 0.4)\n"

# Fix lines 325-327 (0-indexed 324-326)
# Original:
#        unemployment = min(1.0, unemployment + 0.2)
#    unemployment = max(0.0, min(1.0, unemployment))
# Should be:
#        uncertainty = min(1.0, uncertainty + 0.2)
#    uncertainty = max(0.0, min(1.0, uncertainty))
lines[324] = "        uncertainty = min(1.0, uncertainty + 0.2)\n"
lines[325] = "    uncertainty = max(0.0, min(1.0, uncertainty))\n"

with open(file_path, 'w') as f:
    f.writelines(lines)

print("Fixed roi_extraction.py")