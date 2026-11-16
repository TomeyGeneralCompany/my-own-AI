from pathlib import Path
from my_own_ai.web_library import WebLibrary
from my_own_ai.modules import ModuleController, CheckpointStore, default_modules
from my_own_ai.reasoning import Stage


def test_archive_and_shared_lookup(tmp_path, monkeypatch):
    import my_own_ai.web_library as lib
    import my_own_ai.retrieval as retrieval
    def sections(path):
        yield "PDF page 1", "Mangrove forests protect coastal habitats."
    monkeypatch.setattr(lib, "source_sections", sections)
    monkeypatch.setattr(retrieval, "source_sections", sections)
    calls=[]
    def render(url, destination):
        calls.append(url)
        destination.write_bytes(b"%PDF-1.4 fixture")
    archive=WebLibrary(tmp_path/"library",render)
    with ModuleController(CheckpointStore(tmp_path/"memory"),default_modules(),
                          web_library=archive) as controller:
        controller.switch("conversation")
        record=controller.acquire_webpage("https://example.org/mangroves")
        assert (archive.documents/record["pdf"]).exists()
        controller.switch("writing")
        result=controller.search_web_library("mangrove")
        assert result[0]["url"]==record["url"]
        assert result[0]["locator"]=="PDF page 1, passage 1"
    assert WebLibrary(tmp_path/"library").search("mangrove")[0]["pdf_sha256"]==record["sha256"]


def test_core_requests_page_and_receives_reference(tmp_path):
    class Archive:
        acquired=False
        def search(self, query, limit):
            return [{"text":"Mangroves protect coasts."}] if self.acquired else []
        def acquire(self,url):
            assert url=="https://example.org/mangroves"
            self.acquired=True
            return {"url":url}
    class Engine:
        def propose(self,request):
            if not request.context.get("web_references"):
                assert "web_tool" in request.context
                return "WEBPAGE https://example.org/mangroves"
            assert request.context["web_references"][0]["text"]=="Mangroves protect coasts."
            return "Examine the archived evidence."
    with ModuleController(CheckpointStore(tmp_path),default_modules(),Engine(),
                          web_library=Archive()) as controller:
        controller.switch("conversation")
        controller.start("reply","Explain mangroves")
        result=controller.step()
        assert result["stage"]==Stage.CONSTRUCT
        assert result["plan"]=="Examine the archived evidence."


def test_search_then_capture_through_core(tmp_path):
    class Archive:
        acquired=False
        def discover(self, query):
            assert query=="coastal biology"
            return [{"url":"https://example.org/coast","title":"Coastal biology"}]
        def acquire(self,url):
            assert url=="https://example.org/coast"
            self.acquired=True
            return {"url":url}
        def search(self,query,limit):
            return [{"text":"Coastal biology evidence."}] if self.acquired else []
    class Engine:
        def propose(self,request):
            if request.context.get("web_references"):
                return "Use the coastal biology evidence."
            if request.context.get("web_results"):
                return "WEBPAGE "+request.context["web_results"][0]["url"]
            return "WEBSEARCH coastal biology"
    with ModuleController(CheckpointStore(tmp_path),default_modules(),Engine(),
                          web_library=Archive()) as controller:
        controller.switch("conversation")
        controller.start("reply","Explain coastal biology")
        assert controller.step()["plan"]=="Use the coastal biology evidence."


def test_chromium_executes_javascript(tmp_path):
    import pytest
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright
    from threading import Thread
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    from my_own_ai.web_library import render_page
    from my_own_ai.corpus import extract_text
    with sync_playwright() as p:
        if not Path(p.chromium.executable_path).is_file():
            pytest.skip("Chromium not installed")
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type","text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><script>document.body.textContent='Dynamic coastal evidence';</script></body></html>")
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    thread=Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        pdf=tmp_path/"dynamic.pdf"
        render_page(f"http://127.0.0.1:{server.server_port}/",pdf)
        assert "Dynamic coastal evidence" in extract_text(pdf)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
