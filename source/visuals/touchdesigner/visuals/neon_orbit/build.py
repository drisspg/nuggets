from pathlib import Path

PROJECT_ROOT = Path('/Users/drissguessous/Documents/nuggets/source/visuals/touchdesigner')
DIST_DIR = PROJECT_ROOT / 'dist' / 'neon_orbit'
PROJECT_NAME = 'neon_orbit'
WIDTH = 1280
HEIGHT = 720

PIXEL_SHADER = r'''
out vec4 fragColor;

float ring(vec2 p, float r, float width) {
    return 1.0 - smoothstep(width, width + 0.012, abs(length(p) - r));
}

void main() {
    vec2 uv = vUV.st;
    vec2 p = (uv - 0.5) * vec2(16.0 / 9.0, 1.0) * 2.0;
    float t = texture(sTD2DInputs[0], vec2(0.5)).r;

    float a = atan(p.y, p.x);
    float d = length(p);
    float spin = sin(a * 8.0 + t * 1.4) * 0.08;
    float pulse = 0.38 + 0.08 * sin(t * 1.8);

    float r0 = ring(p, pulse + spin, 0.020);
    float r1 = ring(p, 0.72 + 0.04 * sin(a * 5.0 - t * 1.1), 0.016);
    float spokes = pow(max(0.0, sin(a * 12.0 + t * 2.0)), 18.0) * smoothstep(0.15, 0.85, d) * (1.0 - smoothstep(0.85, 1.15, d));
    float core = exp(-d * 4.0) * (0.55 + 0.45 * sin(t * 3.0));

    vec3 cyan = vec3(0.05, 0.85, 1.0);
    vec3 magenta = vec3(1.0, 0.08, 0.72);
    vec3 amber = vec3(1.0, 0.55, 0.08);
    vec3 color = cyan * r0 + magenta * r1 + amber * spokes + vec3(0.2, 0.4, 1.0) * core;
    color += 0.03 * vec3(uv.x, uv.y, 1.0);

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

    time_top = place(root.create(constantTOP, 'time_rgba32f'), -500, 150)
    set_par(time_top, 'outputresolution', 'custom')
    set_par(time_top, 'resolutionw', 1)
    set_par(time_top, 'resolutionh', 1)
    set_par(time_top, 'format', 'rgba32float')
    set_expr(time_top, 'colorr', 'absTime.seconds')
    set_par(time_top, 'colorg', 0)
    set_par(time_top, 'colorb', 0)
    set_par(time_top, 'alpha', 1)

    shader_dat = place(root.create(textDAT, 'neon_orbit_pixel'), -500, -50)
    shader_dat.text = PIXEL_SHADER

    shader = place(root.create(glslTOP, 'neon_orbit_glsl'), -250, 150)
    connect(time_top, shader)
    set_par(shader, 'outputresolution', 'custom')
    set_par(shader, 'resolutionw', WIDTH)
    set_par(shader, 'resolutionh', HEIGHT)
    set_par(shader, 'pixeldat', shader_dat)

    bloom_soften = place(root.create(blurTOP, 'soft_glow_blur'), 0, 150)
    connect(shader, bloom_soften)
    set_par(bloom_soften, 'sizex', 8)
    set_par(bloom_soften, 'sizey', 8)

    glow_gain = place(root.create(levelTOP, 'glow_gain'), 220, 150)
    connect(bloom_soften, glow_gain)
    set_par(glow_gain, 'brightness1', 1.35)
    set_par(glow_gain, 'gamma1', 0.85)

    comp = place(root.create(compositeTOP, 'shader_plus_glow'), 450, 150)
    connect(shader, comp, 0)
    connect(glow_gain, comp, 1)
    set_par(comp, 'operand', 'add')

    out = place(root.create(nullTOP, 'out_neon_orbit'), 680, 150)
    connect(comp, out)
    out.viewer = True
    out.display = True

    rec = place(root.create(moviefileoutTOP, 'recorder'), 680, -80)
    connect(out, rec)
    set_par(rec, 'type', 'movie')
    set_par(rec, 'file', str(DIST_DIR / 'neon-orbit-capture.mov'))
    set_par(rec, 'videocodec', 'prores')

    win = place(root.create(windowCOMP, 'preview_window'), 900, 150)
    set_par(win, 'winop', out.path)
    set_par(win, 'winw', WIDTH)
    set_par(win, 'winh', HEIGHT)

    notes = place(root.create(textDAT, 'README_run_me'), -500, -250)
    notes.text = f'''Neon Orbit TouchDesigner example\n\nOutput TOP: {out.path}\nPreview: pulse {win.path}.par.winopen\nRecord: set {rec.path}.par.record = True, then False to stop\nMovie path: {DIST_DIR / 'neon-orbit-capture.mov'}\nComponent export: {DIST_DIR / (PROJECT_NAME + '.tox')}\nProject save: {DIST_DIR / (PROJECT_NAME + '.toe')}\n'''

    root.save(str(DIST_DIR / f'{PROJECT_NAME}.tox'))
    project.save(str(DIST_DIR / f'{PROJECT_NAME}.toe'))
    return root.path


result = build()
print(f'Built TouchDesigner example at {result}')
