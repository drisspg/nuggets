from pathlib import Path

PROJECT_ROOT = Path('/Users/drissguessous/Documents/nuggets/source/visuals/touchdesigner')
DIST_DIR = PROJECT_ROOT / 'dist' / 'ptq_agent_swarm'
PROJECT_NAME = 'ptq_agent_swarm'
WIDTH = 1280
HEIGHT = 720

PIXEL_SHADER = r'''
out vec4 fragColor;

float hash(float n) {
    return fract(sin(n) * 43758.5453123);
}

float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

float lineGlow(vec2 p, vec2 a, vec2 b, float width) {
    float d = sdSegment(p, a, b);
    return exp(-d * 38.0) * 0.35 + (1.0 - smoothstep(width, width + 0.006, d));
}

float nodeGlow(vec2 p, vec2 c, float r) {
    float d = length(p - c);
    return exp(-d * 18.0) * 0.35 + (1.0 - smoothstep(r, r + 0.006, d));
}

float packet(vec2 p, vec2 a, vec2 b, float phase, float speed) {
    vec2 c = mix(a, b, fract(phase * speed));
    return exp(-length(p - c) * 75.0);
}

float scanBand(vec2 p, float t) {
    float x = fract(t * 0.08) * 2.8 - 1.4;
    return exp(-abs(p.x - x) * 18.0) * 0.08;
}

void addBranch(inout vec3 color, inout float energy, vec2 p, vec2 a, vec2 b, vec3 tint, float reveal, float t, float id) {
    float edgeOn = smoothstep(0.0, 0.22, reveal);
    float edge = lineGlow(p, a, b, 0.006) * edgeOn;
    float pulse = packet(p, a, b, t + id * 0.17, 0.18 + 0.018 * hash(id));
    color += tint * (edge * 0.55 + pulse * 1.45 * edgeOn);
    energy += edge + pulse;
}

void main() {
    vec2 uv = vUV.st;
    vec2 p = (uv - 0.5) * vec2(16.0 / 9.0, 1.0) * 2.0;
    float rawTime = texture(sTD2DInputs[0], vec2(0.5)).r;
    float loopTime = mod(rawTime, 8.0);
    float beat = loopTime / 8.0;
    float reveal = smoothstep(0.05, 0.58, beat);
    float settle = smoothstep(0.62, 0.90, beat);

    vec3 bgTop = vec3(0.015, 0.020, 0.040);
    vec3 bgBottom = vec3(0.040, 0.018, 0.055);
    vec3 color = mix(bgBottom, bgTop, uv.y);
    color += vec3(0.025, 0.035, 0.060) * sin((p.x + p.y) * 8.0 + rawTime * 0.5);
    color += vec3(0.04, 0.07, 0.11) * scanBand(p, rawTime);

    vec3 cyan = vec3(0.08, 0.90, 1.00);
    vec3 magenta = vec3(1.00, 0.10, 0.72);
    vec3 amber = vec3(1.00, 0.62, 0.13);
    vec3 green = vec3(0.18, 1.00, 0.55);
    vec3 violet = vec3(0.50, 0.22, 1.00);

    vec2 dfs0 = vec2(-1.18, -0.72);
    vec2 dfs1 = vec2(-1.05, -0.34);
    vec2 dfs2 = vec2(-1.18, 0.04);
    vec2 dfs3 = vec2(-1.04, 0.40);
    vec2 dfs4 = vec2(-1.18, 0.72);

    float dfs = 0.0;
    dfs += lineGlow(p, dfs0, dfs1, 0.008);
    dfs += lineGlow(p, dfs1, dfs2, 0.008);
    dfs += lineGlow(p, dfs2, dfs3, 0.008);
    dfs += lineGlow(p, dfs3, dfs4, 0.008);
    dfs *= 0.42;
    dfs += packet(p, dfs0, dfs1, beat + 0.00, 1.0) * 0.8;
    dfs += packet(p, dfs1, dfs2, beat + 0.25, 1.0) * 0.8;
    dfs += packet(p, dfs2, dfs3, beat + 0.50, 1.0) * 0.8;
    dfs += packet(p, dfs3, dfs4, beat + 0.75, 1.0) * 0.8;
    color += amber * dfs;

    vec2 root = vec2(-0.25, 0.0);
    vec2 machineA = vec2(0.16, 0.45);
    vec2 machineB = vec2(0.16, 0.0);
    vec2 machineC = vec2(0.16, -0.45);
    vec2 issueA = vec2(0.58, 0.70);
    vec2 issueB = vec2(0.62, 0.43);
    vec2 issueC = vec2(0.62, 0.15);
    vec2 issueD = vec2(0.62, -0.16);
    vec2 issueE = vec2(0.58, -0.47);
    vec2 issueF = vec2(0.54, -0.73);
    vec2 prA = vec2(1.08, 0.54);
    vec2 prB = vec2(1.08, 0.12);
    vec2 prC = vec2(1.08, -0.36);

    float energy = 0.0;
    addBranch(color, energy, p, root, machineA, cyan, reveal, rawTime, 1.0);
    addBranch(color, energy, p, root, machineB, cyan, reveal, rawTime, 2.0);
    addBranch(color, energy, p, root, machineC, cyan, reveal, rawTime, 3.0);
    addBranch(color, energy, p, machineA, issueA, magenta, reveal - 0.10, rawTime, 4.0);
    addBranch(color, energy, p, machineA, issueB, violet, reveal - 0.16, rawTime, 5.0);
    addBranch(color, energy, p, machineB, issueC, magenta, reveal - 0.20, rawTime, 6.0);
    addBranch(color, energy, p, machineB, issueD, violet, reveal - 0.24, rawTime, 7.0);
    addBranch(color, energy, p, machineC, issueE, magenta, reveal - 0.28, rawTime, 8.0);
    addBranch(color, energy, p, machineC, issueF, violet, reveal - 0.32, rawTime, 9.0);
    addBranch(color, energy, p, issueA, prA, green, reveal - 0.38, rawTime, 10.0);
    addBranch(color, energy, p, issueC, prB, green, reveal - 0.44, rawTime, 11.0);
    addBranch(color, energy, p, issueE, prC, green, reveal - 0.50, rawTime, 12.0);

    vec2 nodes[15];
    nodes[0] = root;
    nodes[1] = machineA;
    nodes[2] = machineB;
    nodes[3] = machineC;
    nodes[4] = issueA;
    nodes[5] = issueB;
    nodes[6] = issueC;
    nodes[7] = issueD;
    nodes[8] = issueE;
    nodes[9] = issueF;
    nodes[10] = prA;
    nodes[11] = prB;
    nodes[12] = prC;
    nodes[13] = dfs0;
    nodes[14] = dfs4;

    for (int i = 0; i < 15; i++) {
        float idx = float(i);
        float jitter = (1.0 - settle) * 0.018 * sin(rawTime * (2.8 + hash(idx) * 2.0) + idx * 4.1);
        vec2 c = nodes[i] + vec2(jitter, -jitter * 0.7);
        vec3 tint = mix(cyan, magenta, hash(idx));
        if (i >= 10 && i <= 12) tint = green;
        if (i >= 13) tint = amber;
        float n = nodeGlow(p, c, i == 0 ? 0.045 : 0.032);
        float blink = 0.65 + 0.35 * sin(rawTime * 3.0 + idx);
        color += tint * n * blink;
    }

    float divider = lineGlow(p, vec2(-0.70, -0.82), vec2(-0.70, 0.82), 0.003);
    color += vec3(0.20, 0.28, 0.45) * divider * 0.28;

    float lock = smoothstep(0.82, 0.96, beat) * (1.0 - smoothstep(0.98, 1.0, beat));
    color += green * lock * exp(-abs(length(p - vec2(0.55, 0.0)) - 0.72) * 18.0) * 0.45;
    color += vec3(0.04, 0.12, 0.20) * energy * 0.25;

    color = 1.0 - exp(-color * 1.15);
    color = pow(color, vec3(0.88));
    fragColor = TDOutputSwizzle(vec4(color, 1.0));
}
'''.strip()


def first_par(node, name):
    matches = node.pars(name)
    if matches:
        return matches[0]
    return None


def set_par(node, name, value):
    par = first_par(node, name)
    if par is not None:
        par.val = value
    return par is not None


def set_expr(node, name, expr):
    par = first_par(node, name)
    if par is not None:
        par.expr = expr
    return par is not None


def connect(src, dst, input_index=0):
    dst.inputConnectors[input_index].connect(src.outputConnectors[0])


def clean(parent, child_name):
    existing = parent.op(child_name)
    if existing is not None:
        existing.destroy()


def place(node, x, y):
    node.nodeX = x
    node.nodeY = y
    return node


def build():
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    project_root = op('/project1')
    clean(project_root, PROJECT_NAME)
    root = place(project_root.create(baseCOMP, PROJECT_NAME), 0, 0)

    time_top = place(root.create(constantTOP, 'time_rgba32f'), -520, 180)
    set_par(time_top, 'outputresolution', 'custom')
    set_par(time_top, 'resolutionw', 1)
    set_par(time_top, 'resolutionh', 1)
    set_par(time_top, 'format', 'rgba32float')
    set_expr(time_top, 'colorr', 'absTime.seconds')
    set_par(time_top, 'colorg', 0)
    set_par(time_top, 'colorb', 0)
    set_par(time_top, 'alpha', 1)

    shader_dat = place(root.create(textDAT, 'ptq_agent_swarm_pixel'), -520, -20)
    shader_dat.text = PIXEL_SHADER

    shader = place(root.create(glslTOP, 'ptq_agent_swarm_glsl'), -260, 180)
    connect(time_top, shader)
    set_par(shader, 'outputresolution', 'custom')
    set_par(shader, 'resolutionw', WIDTH)
    set_par(shader, 'resolutionh', HEIGHT)
    set_par(shader, 'pixeldat', shader_dat)

    bloom_soften = place(root.create(blurTOP, 'agent_glow_blur'), 0, 180)
    connect(shader, bloom_soften)
    set_par(bloom_soften, 'sizex', 10)
    set_par(bloom_soften, 'sizey', 10)

    glow_gain = place(root.create(levelTOP, 'agent_glow_gain'), 220, 180)
    connect(bloom_soften, glow_gain)
    set_par(glow_gain, 'brightness1', 1.18)
    set_par(glow_gain, 'gamma1', 0.82)

    comp = place(root.create(compositeTOP, 'swarm_plus_glow'), 450, 180)
    connect(shader, comp, 0)
    connect(glow_gain, comp, 1)
    set_par(comp, 'operand', 'add')

    out = place(root.create(nullTOP, 'out_ptq_agent_swarm'), 690, 180)
    connect(comp, out)
    out.viewer = True
    out.display = True

    rec = place(root.create(moviefileoutTOP, 'recorder'), 690, -80)
    connect(out, rec)
    set_par(rec, 'type', 'movie')
    set_par(rec, 'file', str(DIST_DIR / 'ptq-agent-swarm-capture.mov'))
    set_par(rec, 'videocodec', 'prores')

    win = place(root.create(windowCOMP, 'preview_window'), 930, 180)
    set_par(win, 'winop', out.path)
    set_par(win, 'winw', WIDTH)
    set_par(win, 'winh', HEIGHT)

    notes = place(root.create(textDAT, 'README_run_me'), -520, -240)
    notes.text = f'''ptq Agent Swarm\n\nLeft: one DFS path through a single investigation.\nRight: BFS-style agent fanout across machines, worktrees, issues, and PRs.\nOutput TOP: {out.path}\nPreview: pulse {win.path}.par.winopen\nRecord: python3 tools/record.py --recorder {rec.path} --seconds 8 --mp4 dist/ptq_agent_swarm/ptq-agent-swarm-loop.mp4\nMovie path: {DIST_DIR / 'ptq-agent-swarm-capture.mov'}\nComponent export: {DIST_DIR / (PROJECT_NAME + '.tox')}\nProject save: {DIST_DIR / (PROJECT_NAME + '.toe')}\n'''

    root.save(str(DIST_DIR / f'{PROJECT_NAME}.tox'))
    project.save(str(DIST_DIR / f'{PROJECT_NAME}.toe'))
    return root.path


result = build()
print(f'Built TouchDesigner ptq agent swarm at {result}')
