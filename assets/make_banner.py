"""Generates assets/banner.svg (1400x400). Usage: python assets/make_banner.py [out.svg]
Waveforms are computed so the curves are clean and reproducible."""
import math, random, sys

W, H = 1400, 400
random.seed(7)


def pcg_path(x0, x1, y0, amp, beats, n=900):
    """Phonocardiogram-like trace: S1/S2 bursts (Gabor envelopes) on a flat baseline."""
    pts = []
    period = (x1 - x0) / beats
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        t = (x - x0) / period % 1.0
        v = 0.0
        for c, a, w, f in ((0.12, 1.0, 0.035, 95), (0.47, 0.62, 0.028, 120)):   # S1, S2
            d = t - c
            v += a * math.exp(-(d / w) ** 2) * math.sin(2 * math.pi * f * d)
        v += 0.04 * math.sin(2 * math.pi * 7 * t)
        pts.append((x, y0 - amp * v))
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)


def ecg_path(cx, cy, w, h):
    """Small stylised heartbeat for the logo."""
    xs = [-1, -0.55, -0.42, -0.32, -0.2, -0.08, 0.05, 0.18, 0.32, 0.5, 1]
    ys = [0, 0, -0.18, 0.15, -1.0, 0.75, -0.1, 0, -0.25, 0, 0]
    return "M" + " L".join(f"{cx + x * w / 2:.1f},{cy + y * h / 2:.1f}" for x, y in zip(xs, ys))


# neural-network constellation (right side)
nodes = [(1175, 95), (1250, 70), (1320, 115), (1205, 170), (1290, 195), (1345, 250), (1225, 255), (1160, 230), (1300, 320), (1215, 330)]
nodes = [(round(1255 + (x - 1160) * 0.62), y) for x, y in nodes]
edges = [(0, 1), (1, 2), (0, 3), (3, 4), (2, 4), (4, 5), (3, 6), (6, 7), (4, 6), (5, 8), (6, 9), (8, 9), (1, 3), (7, 9)]

# orbiting "retrieved neighbour" nodes around the logo
LX, LY, LR = 185, 200, 112
orb = [(LX + LR * math.cos(math.radians(a)), LY + LR * math.sin(math.radians(a))) for a in (-62, 18, 118, 202)]

grid = "".join(f'<circle cx="{x}" cy="{y}" r="1.1"/>' for x in range(40, W, 44) for y in range(36, H, 44))

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">
  <title id="t">AuscultRAG</title>
  <desc id="d">Retrieval-Augmented Audio-LLM for Cardiopulmonary Diagnostic Reporting</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#050a18"/>
      <stop offset="0.55" stop-color="#0a1230"/>
      <stop offset="1" stop-color="#140b2e"/>
    </linearGradient>
    <radialGradient id="glowC" cx="0.14" cy="0.5" r="0.42">
      <stop offset="0" stop-color="#22d3ee" stop-opacity="0.28"/>
      <stop offset="1" stop-color="#22d3ee" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="glowP" cx="0.86" cy="0.35" r="0.45">
      <stop offset="0" stop-color="#a855f7" stop-opacity="0.24"/>
      <stop offset="1" stop-color="#a855f7" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="title" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#67e8f9"/>
      <stop offset="0.5" stop-color="#60a5fa"/>
      <stop offset="1" stop-color="#c084fc"/>
    </linearGradient>
    <linearGradient id="ring" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#22d3ee"/>
      <stop offset="0.5" stop-color="#3b82f6"/>
      <stop offset="1" stop-color="#a855f7"/>
    </linearGradient>
    <linearGradient id="wave" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#22d3ee" stop-opacity="0"/>
      <stop offset="0.15" stop-color="#22d3ee" stop-opacity="0.55"/>
      <stop offset="0.6" stop-color="#6366f1" stop-opacity="0.5"/>
      <stop offset="1" stop-color="#a855f7" stop-opacity="0"/>
    </linearGradient>
    <filter id="blur" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="6"/></filter>
    <filter id="soft" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="2.2"/></filter>
    <clipPath id="frame"><rect width="{W}" height="{H}" rx="28" ry="28"/></clipPath>
  </defs>

  <g clip-path="url(#frame)">
    <rect width="{W}" height="{H}" fill="url(#bg)"/>
    <rect width="{W}" height="{H}" fill="url(#glowC)"/>
    <rect width="{W}" height="{H}" fill="url(#glowP)"/>
    <g fill="#8ab4ff" opacity="0.10">{grid}</g>

    <!-- neural constellation -->
    <g stroke="#818cf8" stroke-opacity="0.32" stroke-width="1.2">
      {"".join(f'<line x1="{nodes[a][0]}" y1="{nodes[a][1]}" x2="{nodes[b][0]}" y2="{nodes[b][1]}"/>' for a, b in edges)}
    </g>
    <g>
      {"".join(f'<circle cx="{x}" cy="{y}" r="{4 if i % 3 else 5.5}" fill="{"#c084fc" if i % 2 else "#67e8f9"}" fill-opacity="0.75"/>' for i, (x, y) in enumerate(nodes))}
    </g>

    <!-- phonocardiogram trace -->
    <path d="{pcg_path(0, W, 352, 26, 7)}" fill="none" stroke="url(#wave)" stroke-width="5" filter="url(#blur)" opacity="0.7"/>
    <path d="{pcg_path(0, W, 352, 26, 7)}" fill="none" stroke="url(#wave)" stroke-width="1.6"/>

    <!-- logo: chest-piece ring + heartbeat + retrieved neighbours -->
    <circle cx="{LX}" cy="{LY}" r="{LR}" fill="none" stroke="url(#ring)" stroke-width="1.4" stroke-dasharray="3 7" opacity="0.7"/>
    <g stroke="#67e8f9" stroke-opacity="0.45" stroke-width="1.3">
      {"".join(f'<line x1="{LX}" y1="{LY}" x2="{x:.1f}" y2="{y:.1f}"/>' for x, y in orb)}
    </g>
    <circle cx="{LX}" cy="{LY}" r="74" fill="url(#ring)" opacity="0.35" filter="url(#blur)"/>
    <circle cx="{LX}" cy="{LY}" r="70" fill="#081026" stroke="url(#ring)" stroke-width="5"/>
    <circle cx="{LX}" cy="{LY}" r="56" fill="none" stroke="#3b82f6" stroke-opacity="0.35" stroke-width="1.2"/>
    <path d="{ecg_path(LX, LY + 4, 96, 60)}" fill="none" stroke="#67e8f9" stroke-width="6" stroke-linejoin="round" stroke-linecap="round" filter="url(#soft)" opacity="0.8"/>
    <path d="{ecg_path(LX, LY + 4, 96, 60)}" fill="none" stroke="#e0f7ff" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
    {"".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="#0b1532" stroke="{c}" stroke-width="2.5"/><circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{c}"/>' for (x, y), c in zip(orb, ("#67e8f9", "#60a5fa", "#a78bfa", "#c084fc")))}

    <!-- text -->
    <text x="345" y="178" font-family="Inter, 'Segoe UI', Helvetica, Arial, sans-serif" font-size="96" font-weight="800" letter-spacing="-2" fill="url(#title)">AuscultRAG</text>
    <text x="349" y="226" font-family="Inter, 'Segoe UI', Helvetica, Arial, sans-serif" font-size="26" font-weight="500" fill="#dbe4ff">Retrieval-Augmented Audio-LLM for Cardiopulmonary Diagnostic Reporting</text>

    <!-- feature chips -->
    <g font-family="Inter, 'Segoe UI', Helvetica, Arial, sans-serif" font-size="16" font-weight="600" fill="#cfe9ff">
      <rect x="349" y="256" width="206" height="36" rx="18" fill="#0e1b3d" stroke="#22d3ee" stroke-opacity="0.6"/>
      <text x="452" y="279" text-anchor="middle">Audio-to-Audio Retrieval</text>
      <rect x="567" y="256" width="186" height="36" rx="18" fill="#0e1b3d" stroke="#3b82f6" stroke-opacity="0.7"/>
      <text x="660" y="279" text-anchor="middle">Heart · Lung · Cough</text>
      <rect x="765" y="256" width="236" height="36" rx="18" fill="#0e1b3d" stroke="#a855f7" stroke-opacity="0.7"/>
      <text x="883" y="279" text-anchor="middle">Structured Diagnostic Reports</text>
    </g>
  </g>
  <rect x="0.75" y="0.75" width="{W - 1.5}" height="{H - 1.5}" rx="28" ry="28" fill="none" stroke="#3b82f6" stroke-opacity="0.35" stroke-width="1.5"/>
</svg>
'''
open(sys.argv[1] if len(sys.argv) > 1 else "assets/banner.svg", "w").write(svg)
print("wrote", len(svg), "bytes")
