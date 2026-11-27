"""Execute src chapters and report errors: python tools/check_chapter.py 01_python_calculator ..."""
import sys, jupytext, nbclient, pathlib, traceback
for name in sys.argv[1:]:
    nb = jupytext.read(pathlib.Path("src") / f"{name}.md", fmt="md:myst")
    try:
        nbclient.NotebookClient(nb, timeout=300, kernel_name="python3",
                                resources={"metadata": {"path": "book/part1_python"}}).execute()
        print("OK  ", name)
        import os, nbformat
        if os.environ.get("SAVE_DIR"): nbformat.write(nb, os.path.join(os.environ["SAVE_DIR"], name + ".ipynb"))
    except Exception as e:
        print("FAIL", name, str(e)[-1500:])
