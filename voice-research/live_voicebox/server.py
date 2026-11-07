"""Loopback-only live output demo, separate from the accepted studio server."""
import argparse
import json
import mimetypes
import re
import socket
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
from runtime import Engine,ROOT,RATE,sha

PUBLIC=Path(__file__).resolve().parent/'public'
DUET=ROOT/'evidence/duet-demo/v1/output'


class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'

    def log_message(self,format,*args):
        if str(args[0]).startswith(('POST /api/session','GET /api/stream')):super().log_message(format,*args)

    def setup(self):
        super().setup();self.connection.settimeout(10)
        self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)

    def reply(self,payload,status=200):
        body=json.dumps(payload,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)

    def check_local(self):
        allowed={'127.0.0.1:'+str(self.server.server_port),'localhost:'+str(self.server.server_port)}
        if self.headers.get('Host') not in allowed:raise ValueError('Use the local demo address')
        origin=self.headers.get('Origin')
        if origin is not None and origin not in {'http://'+host for host in allowed}:raise ValueError('Cross-origin requests are not supported')

    def do_GET(self):
        try:
            self.check_local();parsed=urlsplit(self.path)
            if parsed.path=='/api/duet':
                if not (DUET/'manifest.json').is_file():return self.reply({'available':False})
                data=json.loads((DUET/'manifest.json').read_text())
                if not data.get('complete'):return self.reply({'available':False})
                return self.reply({**data,'available':True})
            if parsed.path.startswith('/duet-audio/'):
                name=parsed.path.rsplit('/',1)[-1]
                if parsed.path!='/duet-audio/'+name or name not in ('duet.wav','lead.wav','reply.wav') or not (DUET/'manifest.json').is_file():
                    return self.reply({'error':'Not found'},404)
                data=json.loads((DUET/'manifest.json').read_text())
                entry=next((f for f in data.get('files',[]) if f['file']==name),None)
                if not data.get('complete') or entry is None or sha(DUET/name)!=entry['sha256']:
                    return self.reply({'error':'Duet file is unavailable or changed'},409)
                return self.file(DUET/name,'audio/wav')
            if parsed.path=='/api/catalog':return self.reply(self.server.engine.catalog())
            if parsed.path=='/api/status':
                query=parse_qs(parsed.query);return self.reply(self.server.engine.get(query.get('id',[''])[0]).snapshot())
            if parsed.path=='/api/stream':
                query=parse_qs(parsed.query);return self.stream(self.server.engine.get(query.get('id',[''])[0]))
            if parsed.path.startswith('/sample/'):
                bits=parsed.path.split('/')
                if len(bits)!=4:raise ValueError('Unknown comparison')
                item=self.server.engine.items.get(('speech',bits[2]))
                if item is None:raise ValueError('Unknown comparison')
                choice=next((c for c in item['choices'] if c['choice'].lower()==bits[3]),None)
                if choice is None:raise ValueError('Unknown comparison version')
                path=ROOT/choice['file']
                if sha(path)!=choice['sha256']:raise ValueError('Comparison file changed')
                return self.file(path,'audio/wav')
            relative='index.html' if parsed.path=='/' else parsed.path.lstrip('/')
            if relative not in ('index.html','style.css','app.js','duet.js','voice-pair.js','pcm-player.js','pcm-ring.js','pcm-bytes.js'):
                return self.reply({'error':'Not found'},404)
            return self.file(PUBLIC/relative,mimetypes.guess_type(relative)[0] or 'application/octet-stream')
        except (ValueError,KeyError) as exc:self.reply({'error':str(exc)},400)
        except (BrokenPipeError,ConnectionResetError,socket.timeout):pass

    def file(self,path,content_type):
        body=path.read_bytes();size=len(body);status=200;content_range=None
        if content_type=='audio/wav' and self.headers.get('Range'):
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',self.headers['Range'])
            valid=match is not None and any(match.groups())
            if valid:
                first,last=match.groups()
                if first:
                    start=int(first);end=min(int(last),size-1) if last else size-1
                else:start=max(0,size-int(last));end=size-1
                valid=0<=start<=end<size
            if not valid:
                self.send_response(416);self.send_header('Content-Range',f'bytes */{size}')
                self.send_header('Content-Length','0');self.end_headers();return
            status=206;content_range=f'bytes {start}-{end}/{size}';body=body[start:end+1]
        self.send_response(status);self.send_header('Content-Type',content_type)
        if content_type=='audio/wav':self.send_header('Accept-Ranges','bytes')
        if content_range:self.send_header('Content-Range',content_range)
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)

    def do_POST(self):
        try:
            self.check_local()
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('Send JSON controls')
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=4096:raise ValueError('Invalid request size')
            data=json.loads(self.rfile.read(length))
            if not isinstance(data,dict):raise ValueError('Expected an object')
            if self.path=='/api/session':return self.reply(self.server.engine.create(data).snapshot(),201)
            if self.path=='/api/control':
                identifier=data.pop('id',None);return self.reply(self.server.engine.get(identifier).control(data))
            if self.path=='/api/stop':
                if set(data)!={'id'}:raise ValueError('Invalid stop request')
                session=self.server.engine.get(data['id']);session.stop();return self.reply(session.snapshot())
            return self.reply({'error':'Not found'},404)
        except (ValueError,KeyError,json.JSONDecodeError) as exc:self.reply({'error':str(exc)},400)
        except (BrokenPipeError,ConnectionResetError,socket.timeout):pass

    def stream(self,session):
        with session.lock:
            if session.claimed or session.closed:raise ValueError('This stream is unavailable')
            session.claimed=True
        try:
            self.send_response(200);self.send_header('Content-Type','application/octet-stream')
            self.send_header('Transfer-Encoding','chunked');self.send_header('Cache-Control','no-store')
            self.send_header('X-Sample-Rate',str(RATE));self.send_header('X-Audio-Format','float32-le-mono');self.end_headers()
            start=time.monotonic();sent=0
            while True:
                packet=session.next_chunk()
                if not packet:break
                self.wfile.write(('%x\r\n'%len(packet)).encode()+packet+b'\r\n');self.wfile.flush();sent+=len(packet)//4
                delay=start+sent/RATE-time.monotonic()
                if delay>0:time.sleep(min(delay,.020))
            self.wfile.write(b'0\r\n\r\n');self.wfile.flush()
        except (BrokenPipeError,ConnectionResetError,socket.timeout):pass
        except (ValueError,RuntimeError) as exc:
            self.log_error('Waveform generation ended: %s',exc);self.close_connection=True
        finally:session.stop()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8794);args=parser.parse_args()
    engine=Engine();server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.daemon_threads=True;server.engine=engine
    print('Jaeger Lyra reconstruction demo: http://127.0.0.1:'+str(server.server_port)+'/ · no autoplay',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        for session in engine.sessions.values():session.stop()
        server.server_close()


if __name__=='__main__':main()
