#!/usr/bin/env python3
"""Print installed research environment, without claiming original environment lock."""
import importlib,platform,sys,json
modules=['numpy','pandas','scipy','sklearn','torch','statsmodels','threadpoolctl','openpyxl']
versions={}
for name in modules:
    try:versions[name]=str(getattr(importlib.import_module(name),'__version__','installed/version unknown'))
    except Exception as exc: versions[name]=f'NOT AVAILABLE: {exc}'
print(json.dumps({'python':sys.version,'platform':platform.platform(),'dependencies':versions},indent=2))
