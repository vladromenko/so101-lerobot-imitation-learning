"""Opt-in shim for the project's own scripts: complete /api/scan-hybrid replies with the rim-edge opening.

Enabled only when SO101_OPENING_FALLBACK=1 and this directory is on PYTHONPATH. It touches only the
JSON returned to the calling process; the server and the project's files are not modified.
"""
import os

if os.environ.get('SO101_OPENING_FALLBACK') == '1':
    import io
    import json
    import sys
    import urllib.request as _request
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    _urlopen = _request.urlopen

    def urlopen(target, *args, **kwargs):
        response = _urlopen(target, *args, **kwargs)
        url = getattr(target, 'full_url', target)
        if isinstance(url, str) and url.split('?')[0].endswith('/api/scan-hybrid'):
            try:
                from box_extract.server_compat import complete_scan_record
                record = complete_scan_record(json.loads(response.read()), log=lambda m: print('[opening-fallback]', m, file=sys.stderr))
                response.close()
                return io.BytesIO(json.dumps(record).encode())
            except Exception as exc:                          # never hide the server's own answer
                print('[opening-fallback] not applied:', type(exc).__name__, exc, file=sys.stderr)
                raise
        return response

    _request.urlopen = urlopen
