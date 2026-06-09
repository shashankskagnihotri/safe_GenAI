# Known Limitations

- The current shell did not have diffusers installed, so adapter source inspection is deferred to runtime after environment setup.
- LTX 2.3, Tencent HunyuanVideo, and the native Wan repo did not expose public diffusers `model_index.json` from this environment.
- The dummy adapter validates math and plumbing only; it is not a safety benchmark.
- Offline evaluation hooks are intentionally separate and are never called by the generation runner.
- Full model adapters may need small signature updates as diffusers evolves.

