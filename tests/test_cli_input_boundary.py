import io
import json
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from szl_quant_bench import harness


VALID = b'{"logits":[[1,2]],"provenance":{"source_kind":"SYNTHETIC"}}'


def invoke(source, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["quant-bench", "--input", "source.json"])
    monkeypatch.setattr(harness, "Path", lambda value: source)
    code = harness.main()
    return code, json.loads(capsys.readouterr().out)


class InputFile:
    def __init__(self, body, *, size=None, mode=stat.S_IFREG):
        self.body = body
        self.size = len(body) if size is None else size
        self.mode = mode
        self.read_sizes = []
        self.opened = False

    def stat(self):
        return SimpleNamespace(st_size=self.size, st_mode=self.mode)

    def read_text(self, **kwargs):
        return self.body.decode("utf-8")

    def open(self, mode):
        assert mode == "rb"
        self.opened = True
        owner = self

        class Reader(io.BytesIO):
            def read(self, size=-1):
                owner.read_sizes.append(size)
                return super().read(size)

        return Reader(self.body)


def assert_invalid(outcome):
    code, body = outcome
    assert code == 2
    assert body["state"] == "INVALID"
    assert body["chain"] == []
    assert body["chain_valid"] is False


@pytest.mark.parametrize("body", [
    b'{"logits":[[99]],"logits":[[1,2]],"provenance":{"source_kind":"SYNTHETIC"}}',
    b'{"logits":[[1,2]],"provenance":{"source_kind":"REAL_MODEL_LOGITS"},"provenance":{"source_kind":"SYNTHETIC"}}',
    b'{"logits":[[1,2]],"provenance":{"source_kind":"REAL_MODEL_LOGITS","source_kind":"SYNTHETIC"}}',
    b'{"logits":[[1,2]],"provenance":{"source_kind":"SYNTHETIC","meta":{"run":1,"run":2}}}',
])
def test_ambiguous_json_cannot_mint_a_measured_receipt(body, monkeypatch, capsys):
    assert_invalid(invoke(InputFile(body), monkeypatch, capsys))


def test_growth_after_stat_cannot_bypass_consumed_byte_limit(monkeypatch, capsys):
    source = InputFile(VALID + b" " * 50, size=1)
    monkeypatch.setattr(harness, "MAX_INPUT_BYTES", len(VALID), raising=False)
    assert_invalid(invoke(source, monkeypatch, capsys))
    assert source.read_sizes == [len(VALID) + 1]


@pytest.mark.parametrize("mode", [stat.S_IFIFO, stat.S_IFCHR, stat.S_IFSOCK, stat.S_IFDIR])
def test_nonregular_inputs_are_refused_before_open(mode, monkeypatch, capsys):
    source = InputFile(VALID, mode=mode)
    assert_invalid(invoke(source, monkeypatch, capsys))
    assert source.opened is False


def test_deep_json_is_a_structured_invalid_result(monkeypatch, capsys):
    assert_invalid(invoke(InputFile(b"[" * 2000 + b"0" + b"]" * 2000), monkeypatch, capsys))


@pytest.mark.parametrize("levels", [63, 900])
def test_decodable_deep_provenance_cannot_reach_receipt_serialization(levels, monkeypatch, capsys):
    # Root object + provenance object add two containers beyond the list depth.
    body = (b'{"logits":[[1,2]],"provenance":{"source_kind":"SYNTHETIC","nested":'
            + b"[" * levels + b"0" + b"]" * levels + b"}}")
    assert isinstance(json.loads(body), dict)  # Decoder success is insufficient.
    assert_invalid(invoke(InputFile(body), monkeypatch, capsys))


def test_exact_depth_limit_remains_valid(monkeypatch, capsys):
    body = (b'{"logits":[[1,2]],"provenance":{"source_kind":"SYNTHETIC","nested":'
            + b"[" * 62 + b"0" + b"]" * 62 + b"}}")
    code, report = invoke(InputFile(body), monkeypatch, capsys)
    assert code == 0
    assert report["state"] == "MEASURED"
    assert report["chain_valid"] is True


@pytest.mark.parametrize("extra", [b"NaN", b"Infinity", b"-Infinity", b"1e999"])
def test_nonfinite_values_in_ignored_envelope_fields_are_refused(extra, monkeypatch, capsys):
    source = InputFile(VALID[:-1] + b',"extra":' + extra + b"}")
    assert_invalid(invoke(source, monkeypatch, capsys))


def test_exact_limit_valid_input_keeps_receipt_and_declared_provenance(monkeypatch, capsys):
    source = InputFile(VALID)
    monkeypatch.setattr(harness, "MAX_INPUT_BYTES", len(VALID), raising=False)
    code, body = invoke(source, monkeypatch, capsys)
    assert code == 0
    assert body["state"] == "MEASURED"
    assert body["provenance"] == {"source_kind": "SYNTHETIC"}
    assert body["chain_valid"] is True
    assert len(body["chain"]) == 1


def test_invalid_utf8_is_a_structured_invalid_result(monkeypatch, capsys):
    assert_invalid(invoke(InputFile(b"\xff"), monkeypatch, capsys))


@pytest.mark.parametrize("body, expected", [
    (VALID, "MEASURED"),
    (b'{"logits":[[0]],"logits":[[1,2]],"provenance":{"source_kind":"SYNTHETIC"}}', "INVALID"),
])
def test_real_file_cli_boundary(tmp_path, body, expected):
    source = tmp_path / "input.json"
    source.write_bytes(body)
    result = subprocess.run([sys.executable, "-B", "-m", "szl_quant_bench.harness",
                             "--input", str(source)], capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["state"] == expected
    assert result.returncode == (0 if expected == "MEASURED" else 2)
    if expected == "INVALID":
        assert report["chain"] == []
        assert report["chain_valid"] is False
