import sys

try:
    ipy_str = str(type(get_ipython()))
    if "zmqshell" in ipy_str:
        from tqdm import tqdm_notebook as tqdm
    if "terminal" in ipy_str:
        from tqdm import tqdm
except:
    if sys.stderr.isatty():
        from tqdm import tqdm
    else:

        def tqdm(iterable, **kwargs):
            return iterable
