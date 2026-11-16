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
