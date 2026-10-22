"""Compile the supplied Chinese manuscript. Python standard library only."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

base = Path(__file__).resolve().parent
build = base / 'research_build'
build.mkdir(exist_ok=True)

def executable(name):
    found = shutil.which(name)
    if found:
        return found
    candidate = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData' / 'Local'))) / 'Programs' / 'MiKTeX' / 'miktex' / 'bin' / 'x64' / (name + '.exe')
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(f'{name} not found. Install MiKTeX or TeX Live and add its binaries to PATH.')

try:
    tex = executable('xelatex')
    bib = executable('bibtex')
    tex_args = [tex, '-interaction=nonstopmode', '-halt-on-error', '-file-line-error', '-output-directory=research_build', 'STAND_paper_zh.tex']
    commands = [tex_args, [bib, 'research_build/STAND_paper_zh'], tex_args, tex_args]
    with (build / 'build_console.log').open('w', encoding='utf-8') as log:
        for index, command in enumerate(commands, 1):
            options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
            result = subprocess.run(command, cwd=base, capture_output=True, text=True, encoding='utf-8', errors='replace', **options)
            log.write(f'PASS {index}\n{result.stdout}\n{result.stderr}\n')
            if result.returncode:
                raise RuntimeError(f'Build step {index} failed. Read research_build/build_console.log.')
    shutil.copy2(build / 'STAND_paper_zh.pdf', base / 'STAND_paper_zh.pdf')
    print('Built: STAND_paper_zh.pdf')
except Exception as error:
    print(str(error), file=sys.stderr)
    sys.exit(1)
