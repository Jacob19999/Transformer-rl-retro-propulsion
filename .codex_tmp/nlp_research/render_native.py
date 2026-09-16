"""Use the packaged rasterizer with Word's native PDF export on Windows.

No bundled LibreOffice is supplied by the selected Windows runtime. The Office
export is produced by export_word.ps1; no installed desktop LibreOffice is used.
"""
import importlib.util
import os
import sys
from pathlib import Path
base = Path(r'C:\Transformer-rl-retro-propulsion')
pdf = base / '.codex_tmp/nlp_research/render/aviation_nlp_embedding_project_ideas.pdf'
os.environ['PATH'] = r'C:\Users\tngzj\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin' + os.pathsep + os.environ['PATH']
skill = Path(r'C:\Users\tngzj\.codex\plugins\cache\openai-primary-runtime\documents\26.909.11814\skills\documents\render_docx.py')
spec = importlib.util.spec_from_file_location('render_docx', skill)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
assert pdf.is_file() and pdf.stat().st_size > 0
renderer.convert_to_pdf = lambda *a, **k: (str(pdf), 'Native Microsoft Word PDF export')
sys.argv = [str(skill), str(base/'Plan/NLP/aviation_nlp_embedding_project_ideas.docx'), '--output_dir', str(pdf.parent), '--dpi', '140']
renderer.main()
