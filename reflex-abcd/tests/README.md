# Tests

`test_contracts.py` is the coherence mechanism for the nine-agent build: it
asserts each module defines exactly the public names it owns and that every
implemented signature still matches the frozen one in `src/reflex/contracts.py`.
It passes while modules are stubs (signature checks skip) and keeps passing as
they are implemented.

Still to be written, per spec 10 ("Tests (pytest): ..."). Each belongs to the
agent implementing the module under test:

| test | spec 10 requirement | owner |
|---|---|---|
| `test_data.py` | loader keys present | `data` |
| `test_compile.py` | delex round-trip on 20 fixtures; skeleton extraction on 10 fixtures | `compile` |
| `test_gate.py` | gate routing on synthetic score fixtures, one per reason | `gate` |
| `test_select.py` | candidate mapping | `select` |
| `test_determinism.py` | same context -> same Decision, always | `select` + `gate` |

`test_determinism.py` is called out separately by spec 10 ("Add a test that
asserts it") and by acceptance item 12.
