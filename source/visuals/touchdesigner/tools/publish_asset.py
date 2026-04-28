import argparse
import shutil
from pathlib import Path

TOUCHDESIGNER_ROOT = Path(__file__).resolve().parents[1]
BLOG_SOURCE = TOUCHDESIGNER_ROOT.parents[1]
MEDIA_DIR = BLOG_SOURCE / 'content' / 'media' / 'touchdesigner'


def main():
    parser = argparse.ArgumentParser(description='Copy a rendered TouchDesigner asset into Quartz content media.')
    parser.add_argument('asset', type=Path)
    parser.add_argument('--name')
    args = parser.parse_args()

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    dest = MEDIA_DIR / (args.name or args.asset.name)
    shutil.copy2(args.asset, dest)
    print(dest)
    print('\nEmbed in Markdown:')
    if dest.suffix.lower() in {'.mp4', '.mov', '.webm'}:
        print(f'<video controls loop muted playsinline src="./media/touchdesigner/{dest.name}"></video>')
    else:
        print(f'![[media/touchdesigner/{dest.name}|visual]]')


if __name__ == '__main__':
    main()
