"""CLI client of the one running controller; never opens cameras or servos."""
import argparse
import json
import time
import urllib.error
import urllib.request


def api(path, payload=None):
    request = urllib.request.Request('http://127.0.0.1:8031'+path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={} if payload is None else {'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=5.) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(json.load(exc).get('detail',str(exc))) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('relocate','stow','extract','sort_colors'),default='relocate')
    parser.add_argument('--object',dest='object_id')
    parser.add_argument('--destination')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--confirm-attended',action='store_true',help='accepted for compatibility; current session authorization is already recorded')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--stop',action='store_true')
    args = parser.parse_args()
    try:
        if args.stop:
            print(json.dumps(api('/api/stop',{}),ensure_ascii=False,indent=2));return
        if args.status:
            print(json.dumps(api('/api/status'),ensure_ascii=False,indent=2));return
        api('/api/run',{'mode':args.mode,'object_id':args.object_id,'destination':args.destination,'dry':not args.execute})
        active = True
        while active:
            time.sleep(.25)
            state = api('/api/status')
            active = state['running']
        print(json.dumps({'phase':state['phase'],'detail':state['detail'],'metrics':state['metrics'],'results':state['results']},ensure_ascii=False,indent=2))
        if state['phase'] != 'completed':
            raise SystemExit(1)
    except KeyboardInterrupt:
        api('/api/stop',{})
        raise SystemExit(130)
    except (OSError,RuntimeError) as exc:
        print(str(exc));raise SystemExit(1)


if __name__ == '__main__':
    main()
