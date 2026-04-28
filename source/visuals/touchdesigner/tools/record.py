import argparse
import subprocess
import time
from pathlib import Path

from mcp import execute_td_python

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def convert_to_mp4(mov_path, mp4_path, width, fps, crf):
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            'ffmpeg', '-y', '-i', str(mov_path),
            '-vf', f'scale={width}:-2,fps={fps}',
            '-c:v', 'libx264', '-crf', str(crf),
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an',
            str(mp4_path),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description='Record a TouchDesigner MovieFileOut TOP to web-friendly MP4.')
    parser.add_argument('--recorder', default='/project1/neon_orbit/recorder')
    parser.add_argument('--seconds', type=float, default=6.0)
    parser.add_argument('--mov', type=Path, default=PROJECT_ROOT / 'dist' / 'neon_orbit' / 'neon-orbit-capture.mov')
    parser.add_argument('--mp4', type=Path, default=PROJECT_ROOT / 'dist' / 'neon_orbit' / 'neon-orbit-loop.mp4')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--crf', type=int, default=23)
    args = parser.parse_args()
    args.mov = args.mov.resolve()
    args.mp4 = args.mp4.resolve()

    args.mov.parent.mkdir(parents=True, exist_ok=True)
    args.mov.unlink(missing_ok=True)
    args.mp4.unlink(missing_ok=True)

    print(execute_td_python(f"""
rec = op({args.recorder!r})
rec.par.file = {str(args.mov)!r}
rec.par.record = False
op('/local/time').par.play = True
result = {{'recorder': rec.path, 'movie': rec.par.file.eval(), 'timeline_play': op('/local/time').par.play.eval()}}
"""))
    print(execute_td_python(f"op({args.recorder!r}).par.record = True\nresult = 'recording started'"))
    time.sleep(args.seconds)
    print(execute_td_python(f"op({args.recorder!r}).par.record = False\nresult = 'recording stopped'"))

    deadline = time.time() + 20
    while time.time() < deadline and (not args.mov.exists() or args.mov.stat().st_size == 0):
        time.sleep(0.5)

    if not args.mov.exists() or args.mov.stat().st_size == 0:
        raise SystemExit(f'Recording did not produce a movie at {args.mov}')

    convert_to_mp4(args.mov, args.mp4, args.width, args.fps, args.crf)
    print(f'Wrote {args.mov}')
    print(f'Wrote {args.mp4}')


if __name__ == '__main__':
    main()
