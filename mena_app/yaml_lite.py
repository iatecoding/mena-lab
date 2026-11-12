"""Minimal safe YAML parser for Mena job files and LoT/task registries. Pure Python standard library."""

from __future__ import annotations

import math
import re
from typing import Any

_INT_RE = re.compile(r"^[+-]?(?:0|[1-9][0-9_]*)$")
_HEX_RE = re.compile(r"^[+-]?0x[0-9a-fA-F_]+$")
_FLT_RE = re.compile(r"^[+-]?(?:(?:\d[\d_]*\.\d[\d_]*|\.\d[\d_]+|\d[\d_]*\.)(?:[eE][+-]?\d+)?|\d[\d_]*[eE][+-]?\d+)$")
_ESC = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "0": "\0"}


class YAMLError(ValueError):
    """Raised on malformed or unsupported YAML input."""


def _strip_comment(line: str) -> str:
    in_sq = in_dq = esc = False
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if esc:
            esc = False
        elif in_dq:
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_dq = False
        elif in_sq and ch == "'":
            if i + 1 < n and line[i + 1] == "'":
                i += 2
                continue
            in_sq = False
        elif not in_sq:
            if ch == '"':
                in_dq = True
            elif ch == "'":
                in_sq = True
            elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
                return line[:i].rstrip()
        i += 1
    if in_dq or in_sq:
        raise YAMLError(f"unterminated quoted string in: {line!r}")
    return line.rstrip()


def _unquote_double(s: str) -> str:
    out: list[str] = []
    i, n = 1, len(s) - 1
    while i < n:
        if s[i] != "\\":
            out.append(s[i])
            i += 1
            continue
        i += 1
        if i >= n:
            raise YAMLError(f"trailing backslash in {s!r}")
        ec = s[i]
        if ec in _ESC:
            out.append(_ESC[ec])
            i += 1
        elif ec in "xuU":
            w = {"x": 2, "u": 4, "U": 8}[ec]
            if i + w >= n + 1:
                raise YAMLError(f"invalid unicode escape in {s!r}")
            out.append(chr(int(s[i + 1 : i + 1 + w], 16)))
            i += 1 + w
        else:
            out.append(ec)
            i += 1
    return "".join(out)


def _parse_scalar(token: str) -> Any:
    s = token.strip()
    if not s or s in ("null", "Null", "NULL", "~"):
        return None
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return _unquote_double(s)
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        return s[1:-1].replace("''", "'")
    if s in ("true", "True", "TRUE"):
        return True
    if s in ("false", "False", "FALSE"):
        return False
    low = s.lower()
    if low in (".inf", "+.inf", "-.inf", ".nan"):
        return math.nan if low == ".nan" else (-math.inf if low.startswith("-") else math.inf)
    if _INT_RE.match(s):
        return int(s.replace("_", ""), 10)
    if _HEX_RE.match(s):
        return int(s.replace("_", ""), 16)
    if _FLT_RE.match(s):
        return float(s.replace("_", ""))
    return s


def _scan(s: str):
    """Yield `(index, char, depth, quoted_before)` for characters outside single/double quotes."""
    in_sq = in_dq = esc = quoted = False
    depth = 0
    for i, ch in enumerate(s):
        if esc:
            esc = False
        elif in_dq:
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_dq, quoted = False, True
        elif in_sq:
            if ch == "'":
                in_sq, quoted = False, True
        elif ch == '"':
            in_dq = True
        elif ch == "'":
            in_sq = True
        else:
            if ch in "[{":
                depth += 1
            elif ch in "]}":
                depth -= 1
            yield i, ch, depth, quoted
    if depth != 0 or in_dq or in_sq:
        raise YAMLError(f"unbalanced quotes or brackets in: {s!r}")


def _split_top(s: str, sep: str) -> list[str]:
    cuts = [i for i, ch, depth, _ in _scan(s) if ch == sep and depth == 0]
    bounds = [-1, *cuts, len(s)]
    return [s[bounds[k] + 1 : bounds[k + 1]] for k in range(len(bounds) - 1)]


def _find_key_colon(s: str, in_flow: bool = False) -> int:
    for i, ch, depth, quoted in _scan(s):
        if ch == ":" and depth == 0:
            nxt = s[i + 1] if i + 1 < len(s) else ""
            if nxt == "" or nxt in " \t" or quoted or (in_flow and nxt in ",}]"):
                return i
    return -1


def _parse_flow_or_scalar(s: str) -> Any:
    t = s.strip()
    if not t:
        return None
    if t[0] == "[":
        if not t.endswith("]"):
            raise YAMLError(f"malformed flow sequence: {s!r}")
        inner = t[1:-1].strip()
        if not inner:
            return []
        items = _split_top(inner, ",")
        if items and not items[-1].strip():
            items.pop()
        return [_parse_flow_or_scalar(x) for x in items]
    if t[0] == "{":
        if not t.endswith("}"):
            raise YAMLError(f"malformed flow mapping: {s!r}")
        inner = t[1:-1].strip()
        if not inner:
            return {}
        items = _split_top(inner, ",")
        if items and not items[-1].strip():
            items.pop()
        out: dict[Any, Any] = {}
        for entry in items:
            idx = _find_key_colon(entry, in_flow=True)
            if idx < 0:
                raise YAMLError(f"expected ':' in flow mapping entry {entry!r}")
            out[_parse_scalar(entry[:idx])] = _parse_flow_or_scalar(entry[idx + 1 :])
        return out
    return _parse_scalar(t)


def _flow_depth(s: str) -> int:
    in_sq = in_dq = esc = False
    depth = 0
    for ch in s:
        if esc:
            esc = False
        elif in_dq:
            esc = ch == "\\"
            in_dq = ch != '"'
        elif in_sq:
            in_sq = ch != "'"
        elif ch in ('"', "'"):
            in_dq, in_sq = ch == '"', ch == "'"
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
    return depth


def _prepare_lines(raw_text: str) -> list[tuple[int, str, int]]:
    raw_lines = raw_text.splitlines()
    out: list[tuple[int, str, int]] = []
    i, n = 0, len(raw_lines)
    while i < n:
        raw = raw_lines[i]
        if "\t" in raw[: len(raw) - len(raw.lstrip(" \t"))]:
            raise YAMLError(f"line {i + 1}: tabs are not allowed for YAML indentation")
        chk = raw.lstrip(" ")
        if not chk or chk.startswith("#"):
            i += 1
            continue
        clean = _strip_comment(raw)
        content = clean.lstrip(" ")
        if not content or content in ("---", "..."):
            i += 1
            continue
        indent, start_idx = len(clean) - len(content), i
        while _flow_depth(content) > 0 and i + 1 < n:
            i += 1
            nxt = _strip_comment(raw_lines[i]).strip()
            if nxt:
                content = f"{content} {nxt}"
        out.append((indent, content, start_idx))
        i += 1
    return out


class _Parser:
    def __init__(self, raw_text: str) -> None:
        self.raw_lines = raw_text.splitlines()
        self.lines = _prepare_lines(raw_text)
        self.pos = 0

    def parse(self) -> Any:
        if not self.lines:
            return None
        val = self._block(self.lines[0][0])
        if self.pos < len(self.lines):
            raise YAMLError(f"unexpected trailing content at line {self.lines[self.pos][2] + 1}")
        return val

    @staticmethod
    def _is_seq(text: str) -> bool:
        return text == "-" or text.startswith("- ")

    def _block(self, min_indent: int) -> Any:
        if self.pos >= len(self.lines) or self.lines[self.pos][0] < min_indent:
            return None
        indent, text, _ = self.lines[self.pos]
        if text[0] in "[{":
            self.pos += 1
            return _parse_flow_or_scalar(text)
        if self._is_seq(text):
            return self._sequence(indent)
        if _find_key_colon(text) >= 0:
            return self._mapping(indent)
        self.pos += 1
        return _parse_flow_or_scalar(text)

    def _block_scalar(self, header: str, parent_indent: int, raw_idx: int) -> str:
        folded = header[0] == ">"
        chomp = "-" if "-" in header else ""
        while self.pos < len(self.lines) and self.lines[self.pos][0] > parent_indent:
            self.pos += 1
        stop = self.lines[self.pos][2] if self.pos < len(self.lines) else len(self.raw_lines)
        sub = self.raw_lines[raw_idx + 1 : stop]
        indents = [len(r) - len(r.lstrip(" ")) for r in sub if r.strip()]
        if not indents or min(indents) <= parent_indent:
            return ""
        bi = min(indents)
        chunks = [r[bi:] if len(r) >= bi else "" for r in sub]
        while chunks and chunks[-1] == "":
            chunks.pop()
        body = " ".join(chunks) if folded else "\n".join(chunks)
        return body if chomp == "-" else body + "\n"

    def _value_after_key(self, val_str: str, key_indent: int, raw_idx: int) -> Any:
        v = val_str.strip()
        if v and v[0] in "|>" and v.lstrip("|>+-0123456789") == "":
            return self._block_scalar(v, key_indent, raw_idx)
        if v:
            return _parse_flow_or_scalar(v)
        if self.pos >= len(self.lines):
            return None
        nxt_indent, nxt_text, _ = self.lines[self.pos]
        if nxt_indent > key_indent or (nxt_indent == key_indent and self._is_seq(nxt_text)):
            return self._block(nxt_indent)
        return None

    def _mapping(self, map_indent: int, initial: dict[Any, Any] | None = None) -> dict[Any, Any]:
        out: dict[Any, Any] = {} if initial is None else initial
        while self.pos < len(self.lines):
            indent, text, raw_idx = self.lines[self.pos]
            if indent < map_indent or self._is_seq(text):
                break
            if indent > map_indent:
                raise YAMLError(f"unexpected indentation at line {raw_idx + 1}: {text!r}")
            idx = _find_key_colon(text)
            if idx < 0:
                raise YAMLError(f"expected ':' in mapping at line {raw_idx + 1}: {text!r}")
            key = _parse_scalar(text[:idx])
            self.pos += 1
            out[key] = self._value_after_key(text[idx + 1 :], map_indent, raw_idx)
        return out

    def _sequence(self, seq_indent: int) -> list[Any]:
        items: list[Any] = []
        while self.pos < len(self.lines):
            indent, text, raw_idx = self.lines[self.pos]
            if indent < seq_indent or not (indent == seq_indent and self._is_seq(text)):
                if indent > seq_indent:
                    raise YAMLError(f"unexpected indentation in sequence at line {raw_idx + 1}: {text!r}")
                break
            rest = text[1:].lstrip(" ")
            rest_indent = indent + 1 + (len(text[1:]) - len(rest))
            self.pos += 1
            if not rest:
                items.append(self._block(self.lines[self.pos][0]) if self.pos < len(self.lines) and self.lines[self.pos][0] > seq_indent else None)
            elif rest[0] not in "[{" and _find_key_colon(rest) >= 0:
                idx = _find_key_colon(rest)
                d: dict[Any, Any] = {_parse_scalar(rest[:idx]): self._value_after_key(rest[idx + 1 :], rest_indent, raw_idx)}
                if self.pos < len(self.lines) and self.lines[self.pos][0] == rest_indent and not self._is_seq(self.lines[self.pos][1]):
                    self._mapping(rest_indent, initial=d)
                items.append(d)
            elif rest[0] in "|>" and rest.lstrip("|>+-0123456789") == "":
                items.append(self._block_scalar(rest, seq_indent, raw_idx))
            else:
                items.append(_parse_flow_or_scalar(rest))
        return items


def safe_load(source: Any) -> Any:
    """Parse a safe subset of YAML from a string or readable stream into standard Python objects."""
    text = source.read() if hasattr(source, "read") else source
    if not isinstance(text, str):
        raise YAMLError("expected a YAML string or text stream")
    return _Parser(text).parse()
