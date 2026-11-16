# Shared web-to-PDF library

All active modules use the controller's same persistent `web-library` directory under module memory. This is reference material, not training data or an alternative reasoning engine.

Install the system `wkhtmltopdf` package and the existing Python books extra (`pip install -e '.[books]'`). Capture known HTTP(S) URLs. The renderer is conventional software; JavaScript is disabled in this initial implementation. Dynamic/login-only pages and bot challenges may not render; image-only PDFs are retained but do not provide searchable text. There is no search-engine discovery service or access-control bypass.

Owner session commands: 
- `{"command":"acquire_webpage","url":"https://example.org/"}`
- `{"command":"search_web_library","query":"relevant terms","limit":3}`

Core proposal protocol: `WEBPAGE https://example.org/` is an acquisition action, not a completed proposal. The controller renders and archives the page, records the URL, timestamp and PDF digest, rebuilds the existing SQLite index, and offers matching passages to the proposing model. Additional distinct requests can follow. The original PDF persists across module changes and restarts. No external AI is called. 

Current checks use a fixture renderer; a real network capture and autonomous action selection by the trained BPE Core remain to be demonstrated. Web passages are advisory data, not automatically validated premises in existing module evaluators. No changes to weights or staged training.
