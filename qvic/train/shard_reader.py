from __future__ import annotations

import io
import os

HOUND_PREFIX = "shareVideoGPTV/frames/all_frames/"


class ShardReader:
    def __init__(self, root: str):
        import pandas as pd
        self.root = root
        idx = pd.read_parquet(os.path.join(root, "metadata", "shard_index.parquet"))
        self._idx = {m: (s, o, n) for m, s, o, n in
                     zip(idx.member, idx.shard, idx.offset, idx.nbytes)}
        self._kind = dict(zip(idx.member, idx.kind))
        self._scene: dict[str, list[str]] = {}
        for m, k in zip(idx.member, idx.kind):
            if k == "frame":
                self._scene.setdefault(m.split("/", 1)[0], []).append(m)
        for v in self._scene.values():
            v.sort()
        self._fh: dict[str, io.BufferedReader] = {}
        self._pid = os.getpid()


    def _handle(self, shard: str, kind: str) -> io.BufferedReader:
        if os.getpid() != self._pid:
            self._fh.clear()
            self._pid = os.getpid()
        fh = self._fh.get(shard)
        if fh is None:
            sub = "video" if kind == "video" else "frame"
            fh = open(os.path.join(self.root, "shards", sub, shard), "rb", buffering=0)
            self._fh[shard] = fh
        return fh

    def _resolve(self, key: str) -> str:
        """Annotation `video` field -> index member key."""
        return key[len(HOUND_PREFIX):] if key.startswith(HOUND_PREFIX) else key


    def __contains__(self, key: str) -> bool:
        k = self._resolve(key)
        return k in self._idx or k in self._scene

    def kind_of(self, key: str) -> str:
        """'video' or 'scene' -- which accessor this key needs."""
        k = self._resolve(key)
        if k in self._scene:
            return "scene"
        if k in self._idx:
            return self._kind[k]
        raise KeyError(key)

    def open(self, key: str):
        """Whatever this key represents: a decord VideoReader or a frame list."""
        return self.scene_frames(key) if self.kind_of(key) == "scene" \
            else self.video_reader(key)

    def path_of(self, key: str) -> str | None:
        """Filesystem path if the member is stored loose, else None."""
        member = self._resolve(key)
        shard, _, _ = self._idx[member]
        return os.path.join(self.root, "videos_large", member) if shard == "" else None

    def read(self, key: str) -> bytes:
        """Raw bytes of one member (one pread for packed members)."""
        member = self._resolve(key)
        shard, offset, nbytes = self._idx[member]
        if shard == "":
            with open(os.path.join(self.root, "videos_large", member), "rb") as fh:
                return fh.read()
        fh = self._handle(shard, self._kind[member])
        return os.pread(fh.fileno(), nbytes, offset)

    def video_reader(self, key: str, **kw):
        """decord.VideoReader for a video member; loose files go by path."""
        from decord import VideoReader, cpu
        kw.setdefault("ctx", cpu(0))
        p = self.path_of(key)
        return VideoReader(p, **kw) if p else VideoReader(io.BytesIO(self.read(key)), **kw)

    def scene_members(self, key: str) -> list[str]:
        """Sorted frame members of a llava_hound scene."""
        scene = self._resolve(key)
        try:
            return self._scene[scene]
        except KeyError:
            raise KeyError(f"unknown scene {scene}") from None

    def scene_frames(self, key: str) -> list:
        """Decoded PIL frames of a llava_hound scene."""
        from PIL import Image
        return [Image.open(io.BytesIO(self.read(m))).convert("RGB")
                for m in self.scene_members(key)]

    def close(self) -> None:
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()
