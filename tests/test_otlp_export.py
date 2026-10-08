"""The OTLP export path really delivers spans: a local OTLP/HTTP receiver decodes what the exporter sends."""
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest, ExportTraceServiceResponse

from agent import telemetry


def test_spans_reach_an_otlp_http_endpoint():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            req = ExportTraceServiceRequest()
            req.ParseFromString(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, req))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.end_headers()
            self.wfile.write(ExportTraceServiceResponse().SerializeToString())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    provider = telemetry.make_provider(f"http://127.0.0.1:{server.server_port}")
    sp = provider.get_tracer("t").start_span("llm.extract")
    sp.set_attribute("gen_ai.request.model", "gpt-4o-mini")
    sp.set_attribute("gen_ai.usage.input_tokens", 123)
    sp.end()
    provider.force_flush()
    provider.shutdown()
    server.shutdown()

    assert received and received[0][0] == "/v1/traces"
    span = received[0][1].resource_spans[0].scope_spans[0].spans[0]
    attrs = {kv.key: kv.value for kv in span.attributes}
    assert span.name == "llm.extract" and attrs["gen_ai.request.model"].string_value == "gpt-4o-mini"
    assert attrs["gen_ai.usage.input_tokens"].int_value == 123
