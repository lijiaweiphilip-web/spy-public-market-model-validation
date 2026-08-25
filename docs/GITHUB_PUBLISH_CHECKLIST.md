# GitHub publishing checklist

- [ ] Run `pytest` and `ruff check src tests` locally.
- [ ] Run the full reference pipeline using a local raw-data snapshot.
- [ ] Run `spy-validate validate --run-dir <run_dir>`.
- [ ] Confirm that raw market data and point-level private evidence are excluded from the public upload.
- [ ] Upload only paths listed in `PUBLISH_ALLOWLIST.txt`.
- [ ] Confirm no OpenReview PDFs, submission IDs, reviewer files, credentials, NTU identifiers or private application files are present.
- [ ] Create the repository as **private** first.
- [ ] Confirm GitHub Actions passes on all configured Python versions.
- [ ] Review the rendered README and links.
- [ ] Change visibility to public only after explicit user approval.
- [ ] Add the final public repository link to the CV only after the public page and Actions are verified.
