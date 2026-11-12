# Official Conda binary validation — 2026-11-12

Both CPU and CUDA profiles passed all 10 recorded steps. See `cpu/result.json`
and `cuda/result.json`; individual commands and logs are adjacent.

- New, separate Conda environments; official dglteam Python 3.11 binaries for
  PyTorch 2.3, CPU and CUDA 12.1. DGL provenance is retained for each profile.
- SIGMMA built as a regular wheel and installed with dependency resolution;
  imports resolve inside the selected environment, not the source checkout.
- PyTorch 2.3.1, torchvision 0.18.1, NumPy 1.26.4; DGL 2.4.0 / 2.4.0+cu121.
- Both `pip check` results: no broken requirements.
- Core imports, GraphConv and LSTM SAGEConv forward/backward, edge softmax,
  batch/unbatch, edge subgraph: passed. CUDA execution used the available A100.
- All four installed CLI entrypoints passed `--help`.
- Tested wheel SHA-256 (both profiles):
  `b49d4745c33b24e62f5e0730c9ec767f7432d19c245a28f0c6254e03f7f7e2b4`.

## Environment repair

Initial CPU runtime and CUDA precompute CLI checks failed because SciPy loaded
the system C++ runtime, which lacks `CXXABI_1.3.15`. Prepending the environment's
`lib` directory to `LD_LIBRARY_PATH` fixed both. Initial logs are preserved in
each profile's `initial_attempt/`. Package/model code and metadata were not
modified. The passing environment settings are recorded in the result JSON.

## TLS diagnosis and limits

During the earlier diagnosis on the same date, local, Google and Cloudflare
DNS all returned `104.219.250.37` for `data.dgl.ai`. Both default TLS and TLS 1.2
connections closed after ClientHello, before receipt of a server certificate.
The domain's nameservers pointed to `ns1/2.lander.d.parity.domains`; HTTP returned
404. This supports a domain/hosting problem, not a SIGMMA dependency failure;
the exact upstream administrative cause is unconfirmed. Certificate checking
was never disabled. Official Conda binaries provide a working alternative.

This does not validate DGL's unavailable pip wheel host, SIGMMA's published
PyPI artifact, model-weight downloads, full training, gradient equivalence,
checkpoint accuracy, or real-data inference. Documentation was updated after
the tested wheel build; the wheel hash identifies the exact tested artifact.
