from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

from algo.ml_pred import predictDisease
import json


class MyRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = parse_qs(post_data.decode('utf-8'))
        arrayofSympt = data.get('addedSymptoms', [])
        arrayofSympt = ",".join(arrayofSympt)
        predicted = predictDisease(arrayofSympt)
        print(arrayofSympt)
        print("---webserver debug--" , " predicted:", predicted)
        # Set the headers for CORS
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

        # Send the response body
        self.wfile.write(json.dumps(predicted).encode('utf-8'))


if __name__ == '__main__':
    PORT = 8080
    server_address = ('localhost', PORT)
    httpd = HTTPServer(server_address, MyRequestHandler)
    print(f'Starting web server on port {PORT}')
    httpd.serve_forever()