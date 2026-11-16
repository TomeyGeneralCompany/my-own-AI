# Shared web-to-PDF library

Every active module uses the controller's persistent web-library directory. Sources are reference material, not changes to training or an alternative reasoning engine.

Install: `pip install -e '.[web]'` and `python -m playwright install --with-deps chromium`.

Owner session commands:
- `{"command":"discover_webpages","query":"coastal biology","limit":5}`
- `{"command":"acquire_webpage","url":"https://example.org/"}`
- `{"command":"search_web_library","query":"relevant terms","limit":3}`

Core proposal actions: WEBSEARCH followed by query terms returns ordinary Google search result titles and URLs. WEBPAGE followed by an HTTP(S) URL opens it in JavaScript-enabled headless Chromium and prints the rendered page to PDF. The controller archives the document, original and final URL, title, capture time and PDF digest, builds its searchable index and returns source passages. The Core decides which links and queries to use; successive distinct requests are supported. Chromium sessions are closed after each operation. No external AI is involved.

Search markup changes, provider challenges, access restrictions, and pages requiring interaction or authentication can prevent acquisition. No bypass is provided. Scanned PDFs require separate OCR. Search and capture tests include fixture round trips and a local JavaScript page (requires installed Chromium). Live Google discovery and genuine trained Core action selection remain unverified. Web passages are advisory reference data rather than automatically validated premises in existing module evaluators. No training changes.
