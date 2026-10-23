# The FHERMA image

`Dockerfile.fherma` is perseus-core as proposed to [FHERMA](https://fherma.io): CUDA 13.0.3, Rust 1.99.1, the kernels
built for `sm_89` and `sm_120` (FHERMA's L40S and RTX PRO 6000 Blackwell). `build-fherma.sh` is its build, and runs on
a host too.

| path | what |
| --- | --- |
| `/opt/perseus/crates` | the Rust workspace; an answer path-depends on `pc-gpu`, `pc-core`, ... |
| `/opt/cargo` | the registry warmed with every crate of `Cargo.lock`: answers build offline |
| `/opt/perseus/venv` | Python 3.12 with `perseus` and `perseus._pc`, first on `PATH` |
| `/opt/perseus/bin` | `pc-kernels-selftest --no-bench` and `pc-gpu-smoke`, the GPU checks |

```bash
docker build -f docker/Dockerfile.fherma -t perseus-core .
docker run --rm --gpus all perseus-core pc-gpu-smoke
```

The workflow `.github/workflows/fherma-image.yml` pushes it to `ghcr.io/<owner>/perseus-core`; a `v*` tag publishes
that version, and FHERMA pins the digest from the run summary.
